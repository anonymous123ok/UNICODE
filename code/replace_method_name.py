import re


def replace_function_name_multiline(code: str, new_name: str = "methodName") -> str:
    
    pattern = r'([a-zA-Z_][\w\s\*\&]*?)\s+([a-zA-Z_]\w*)\s*\((.*?)\)\s*\{'

    def repl(match):
        return f"{match.group(1)} {new_name}({match.group(3)})" + " {"

    # return re.sub(pattern, repl, code, count=1, flags=re.DOTALL)
    return re.sub(pattern, repl, code,flags=re.DOTALL)


result = replace_function_name_multiline(code)
print(result)