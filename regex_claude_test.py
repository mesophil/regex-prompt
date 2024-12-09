import re


def convert_to_safe_regex(input_string):
    """Convert a string to a safe, usable regex pattern
    
    Args:
        input_string (str): Input string containing a regex-like pattern
    
    Returns:
        str: A safe, compilable regex pattern
    """
    # First, handle escaped characters that might be problematic
    def fix_escapes(pattern):
        # Replace specific problematic escape sequences
        pattern = pattern.replace('\\', '\\\\')
        return pattern
    
    # Unescape special regex characters that are intentionally part of the pattern
    def unescape_regex_chars(pattern):
        # Preserve intentional grouping and alternation
        pattern = pattern.replace('\\(', '(')
        pattern = pattern.replace('\\)', ')')
        pattern = pattern.replace('\\|', '|')
        
        return pattern
    
    # First, fix escapes
    fixed_pattern = fix_escapes(input_string)
    
    # Then unescape intentional regex characters
    safe_pattern = unescape_regex_chars(fixed_pattern)
    
    return safe_pattern

def test_regex_converter():
    # Test cases with various regex-like strings
    test_patterns = [
        '(hello|greetings|salutations|it nice to meet you|good morning|good evening|good afternoon|how are you doing|how do you)',
        'test\\something special',
        'line1 line2',
        'regex*special.chars',
        'backslash\\test',
        '\b(hello|world)\b',
        'pattern\with\backslashes'
    ]
    
    for pattern in test_patterns:
        print("Original pattern:", repr(pattern))
        safe_regex = convert_to_safe_regex(pattern)
        print("Safe regex:", repr(safe_regex))
        
        # Demonstrate usability with re.findall()
        try:
            test_strings = [
                'hello world',
                'it nice to meet you',
                'good morning',
                'how are you doing',
                'test something special',
                'line1   line2',
                'regex is special',
                'backslash test',
                'hello',
                'pattern with backslashes'
            ]
            
            for test_string in test_strings:
                matches = re.findall(safe_regex, test_string)
                if matches:
                    print(f"  Matches in '{test_string}': {matches}")
        except Exception as e:
            print(f"  Error: {e}")
        print()

# Run the test function
if __name__ == '__main__':
    test_regex_converter()