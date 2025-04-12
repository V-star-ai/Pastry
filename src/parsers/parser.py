from probably.pgcl import parse_pgcl
from probably.pgcl.ast import *
from src.analysis.guard_analysis import GuardExpr
from src.transformers.bounded import convert_bounded_pcp
from src.transformers.condbounded import convert_condbounded_pcp
from src.transformers.constant import check_const_guard, convert_const_pcp
from src.transformers.monotone import check_mono_pcp, convert_mono_pcp
from sympy import parse_expr
import logging
import re


logger = logging.getLogger('pastry')


def remove_comments_from_code(input_string):
    """
    Remove all single-line comments (starting with '#') from the given program string.

    :param input_string: The full program as a string, possibly containing line comments.
    :return: A string with all line comments removed.
    """
    lines = input_string.splitlines()
    cleaned_lines = [line.split('#', 1)[0].rstrip() for line in lines]
    return '\n'.join(cleaned_lines)



def parse_program_info(input_string):
    """
    Parse the metadata annotation block (/*@...@*/) from the program header.
    """
    # Search for the annotation block of the form /*@ ... @*/
    match = re.search(r"/\*@([\s\S]*?)@\*/", input_string)
    if not match:
        return None, input_string # No annotation found; return original code
    
    # Split by delimiters: ',', '[', and ']', then trim and remove empty entries
    content = match.group(1).strip()
    remaining_part = input_string.replace(match.group(0), "", 1).strip()
    parts = [p.strip() for p in re.split(r'[\[\],]', content) if p.strip()]
    category = parts.pop(0)  
    
    if category == 'Bounded':
        # If variable count not divisible by 3, treat the first part as unbounded variable
        if len(parts) % 3 != 0:
            M_str = parts.pop(0)
        else:
            M_str = None

        var_bound_dict = {}
        var_comp_dict = {}
        
        # Extract variable annotations and compute derived information
        for i in range(0, len(parts), 3):
            var_name = parts[i]
            lower = int(parts[i+1])
            upper = int(parts[i+2])
            comp = -lower if lower < 0 else 0
            var_bound_dict[var_name] = comp + upper + 1
            var_comp_dict[var_name] = comp
        var_bound_dict = dict(sorted(var_bound_dict.items(), key=lambda x: x[1]))
        return category, remaining_part, var_comp_dict, var_bound_dict, M_str
    
    elif category == 'CondBounded':
        # Extract variable annotations and compute derived information
        central_var = parts.pop(0)
        var_abc_info = {}
        for i in range(0, len(parts), 5):
            var_name = parts[i]
            var_abc_info[parts[i]] = (int(parts[i+1]), int(parts[i+2]), int(parts[i+3]), int(parts[i+4]))
        return category, remaining_part, var_abc_info, central_var
    
    else:
        raise TypeError(f"Unsupported type: {category}")


def split_string(input_string):
    """
    Split the program string into two parts: the initial variable declarations and the remaining program body. 
    Also extract the initialized values of the declared variables.

    :param input_string: The program string, including variable declarations.
    :return: A tuple containing:
             - remaining_part (str): the program string with declarations removed.
             - variables_dict (dict): a mapping from variable names to their initialized values.
    """
    
    # Remove leading whitespace from the input
    input_string = input_string.lstrip()
    
    # Pattern to match one or more variable declarations at the beginning of the string.
    pattern = r'^\s*(int\s+[a-zA-Z_]\w*\s*=\s*-?\d+\s*(?:;|\r?\n)\s*)+'
    # Pattern to capture variable name and value from each declaration
    var_value_pattern = r'int\s+(?P<var>[a-zA-Z_]\w*)\s*=\s*(?P<value>-?\d+)\s*(?:;|\r?\n)'

    # Try to match the initial declaration block from the input
    match = re.match(pattern, input_string)

    declaration_part = ''
    remaining_part = input_string
    variables_dict = {}

    if match:
        # Extract the declaration block
        declaration_part = match.group(0)
        
        # Remaining part is the input after the declaration block
        remaining_part = input_string[len(declaration_part):]

        # Extract the initialized values of the declared variables
        for var_match in re.finditer(var_value_pattern, declaration_part):
            var_name = var_match.group('var')
            var_value = int(var_match.group('value'))
            variables_dict[var_name] = var_value

    return remaining_part.lstrip(), variables_dict


def replace_guards(code):
    """
    Replace all guard expressions in `if` and `while` statements with unique labels (e.g., "guard_0"),
    and collect the set of variables that appear in any guard condition.
    This transformation is necessary to ensure that the resulting program string can be parsed
    by the `probably` library.
    
    :param code: The program string containing guard expressions.
    :return: A tuple containing:
             - modified_code (str): the program string with guard expressions replaced by labels.
             - replacement_map (dict): mapping from each guard label to its parsed expression.
             - meaningful_vars (set): set of variables used in all guard expressions.
    """
    
    # Match 'if (...) {' or 'while (...) {' blocks and capture the inner guard condition
    pattern = re.compile(r'\b(if|while)\s*\(\s*(.*?)\s*\)\s*{', re.DOTALL)
    replacement_map = {}
    meaningful_vars = set()
    replacement_counter = 0

    def replacement_function(match):
        nonlocal replacement_counter
        statement_type = match.group(1)
        condition_content = match.group(2)
        replacement_key = f'guard_{replacement_counter}'
        
        # Parse the condition using SymPy's symbolic expression parser
        sp_expr = parse_expr(condition_content)
        replacement_map[replacement_key] = sp_expr
        
        # Collect all variables used in the guard expression
        meaningful_vars.update(str(var) for var in sp_expr.free_symbols)
        replacement_counter += 1
        
        # Replace original guard with a unique label
        return f'{statement_type}({replacement_key}){{'

    # Apply guard replacement to the entire program string
    modified_code = pattern.sub(replacement_function, code)
    return modified_code, replacement_map, meaningful_vars



def remove_redundant_instructions(syntax_tree, meaningful_vars):
    """
    Recursively remove redundant assignment and control instructions that do not affect 
    the termination behavior of the program.

    Specifically, this includes:
        - Skip statements
        - Assignments to variables that never appear in any guard condition
        - Identity updates, e.g., x := x + 0 or x := x - 0
        - Trivial assignments, e.g., x := x + 0

    :param syntax_tree: The program syntax tree to be simplified.
    :param meaningful_vars: Set of variables that appear in guard conditions.
    :return: 
    """
    if isinstance(syntax_tree, list): 
        i = 0
        while i < len(syntax_tree):
            
            if isinstance(syntax_tree[i],SkipInstr):
                del syntax_tree[i]
                continue
                
            if isinstance(syntax_tree[i], AsgnInstr):
                if syntax_tree[i].lhs not in meaningful_vars:
                    del syntax_tree[i]
                    continue
                
                if isinstance(syntax_tree[i].rhs, VarExpr):
                    del syntax_tree[i]
                    continue 
                else:
                    if isinstance(syntax_tree[i].rhs.lhs, VarExpr):
                        if syntax_tree[i].rhs.rhs.value == 0:
                            del syntax_tree[i]
                            continue
                    else:
                        if syntax_tree[i].rhs.lhs.value == 0:
                            del syntax_tree[i]
                            continue
                        else:
                            syntax_tree[i].rhs = BinopExpr(operator=syntax_tree[i].rhs.operator, lhs=syntax_tree[i].rhs.rhs, rhs=syntax_tree[i].rhs.lhs)
                    
            remove_redundant_instructions(syntax_tree[i], meaningful_vars)
            i += 1
            
    elif isinstance(syntax_tree, WhileInstr):
        remove_redundant_instructions(syntax_tree.body, meaningful_vars)
        
    elif isinstance(syntax_tree, IfInstr):
        remove_redundant_instructions(syntax_tree.true, meaningful_vars)
        remove_redundant_instructions(syntax_tree.false, meaningful_vars)
        
    elif isinstance(syntax_tree, ChoiceInstr):
        remove_redundant_instructions(syntax_tree.lhs, meaningful_vars)
        remove_redundant_instructions(syntax_tree.rhs, meaningful_vars)


def reverse_replace_instruction(syntax_tree, replacement_map):
    """
    Reverse the earlier transformation that replaced complex guard expressions 
    with labels such as "guard_0", "guard_1", etc.

    :param syntax_tree: The program syntax tree to be updated.
    :param replacement_map: A dictionary mapping guard label (e.g., "guard_0") 
                            to its original symbolic expression.
    :return: 
    """
    if isinstance(syntax_tree, list):
        for item in syntax_tree:
            reverse_replace_instruction(item, replacement_map)

    elif isinstance(syntax_tree, WhileInstr):
        if isinstance(syntax_tree.cond, VarExpr) and bool(re.match(r'^guard_\d+$', syntax_tree.cond.var)):
            syntax_tree.cond = GuardExpr(replacement_map[syntax_tree.cond.var])
        reverse_replace_instruction(syntax_tree.body, replacement_map)

    elif isinstance(syntax_tree, IfInstr):
        if isinstance(syntax_tree.cond, VarExpr) and bool(re.match(r'^guard_\d+$', syntax_tree.cond.var)):
            syntax_tree.cond = GuardExpr(replacement_map[syntax_tree.cond.var])
        reverse_replace_instruction(syntax_tree.true, replacement_map)
        reverse_replace_instruction(syntax_tree.fals, replacement_map)

    elif isinstance(syntax_tree, ChoiceInstr):
        reverse_replace_instruction(syntax_tree.lhs, replacement_map)
        reverse_replace_instruction(syntax_tree.rhs, replacement_map)


def parse_pcp(pcp_str):
    """
    Parse a probabilistic counter program string into a syntax tree.
    
    Automatically detects and handles 1-d, Constant, and Monotone PCPs. 
    For Bounded or Conditionally Bounded PCPs, requires annotation via /*@...@*/.

    :param pcp_str: The raw probabilistic counter program string.
    :return: The input program parsed into a normalized syntax tree object.
    """
    logger.info("Preprocessing started.")
    
    # Remove inline and block comments from the input program
    pcp_str = remove_comments_from_code(pcp_str)
    
    # Extract metadata annotations of the form /*@...@*/ from the program
    meta_info = parse_program_info(pcp_str)
    
    # Separate initial variable declarations from the program body
    pcp_remaining, pcp_dict = split_string(meta_info[1])
    
    # Parse the program with transformed guard expressions
    sd_pgcl_str, replacement_map, meaningful_vars = replace_guards(pcp_remaining)
    sd_pgcl_prog = parse_pgcl(sd_pgcl_str)
    
    # Filter out unused variables and redundant instructions from the program
    sd_pgcl_prog.variables = {key: value for key, value in pcp_dict.items() if key in meaningful_vars}
    if not sd_pgcl_prog.variables:
        sd_pgcl_prog.variables['x'] = 0
    remove_redundant_instructions(sd_pgcl_prog.instructions, meaningful_vars)
    
    if len(pcp_dict)>1:
        if meta_info[0]:
            if meta_info[0] == 'Bounded':
                logger.info(f"Program classified as Bounded ({len(pcp_dict)}-d PCP).")
                convert_bounded_pcp(sd_pgcl_prog, replacement_map, meta_info[2], meta_info[3], meta_info[4])
            elif meta_info[0] == 'CondBounded':
                logger.info(f"Program classified as Conditionally Bounded ({len(pcp_dict)}-d PCP).")
                convert_condbounded_pcp(sd_pgcl_prog, replacement_map, meta_info[2], meta_info[3])
        else:
            is_valid, ct_guard_dict, bench_coeff_dict = check_const_guard(replacement_map)
            if is_valid:
                logger.info(f"Program classified as Constant ({len(pcp_dict)}-d PCP).")
                convert_const_pcp(sd_pgcl_prog, ct_guard_dict, bench_coeff_dict)
            else:
                is_valid, var_trend_dict, info_dict = check_mono_pcp(sd_pgcl_prog, replacement_map)
                if is_valid:
                    logger.info(f"Program classified as Monotone ({len(pcp_dict)}-d PCP).")
                    convert_mono_pcp(sd_pgcl_prog, replacement_map, var_trend_dict, info_dict)
                else:
                    logger.error("Failed to classify input program as any PCP category supported by Pastry.")
                    raise ValueError(
                        "Input program could not be classified as 1-d PCP, Constant k-d PCP, or Monotone k-d PCP, "
                        "and no valid annotation was detected."
                        "Please ensure the program conforms to a supported PCP class or includes a proper annotation block."
                    )
    else:
        logger.info("Program classified as 1-d PCP.")
        reverse_replace_instruction(sd_pgcl_prog.instructions, replacement_map)
    
    logger.info("Preprocessing completed.")
    return sd_pgcl_prog
