import re

# The following regex will match any line containing 'renderfig', 'default_mpl_style'
# or '[...]':
# - '^' matches the start of a line
# - '.*' matches any character (except for line terminators) zero or more times
# - '$' matches the end of a line
# - '\n?' optionally matches the newline character at the end of the line
regex = r'^.*(renderfig|default\_mpl\_style|\[\.\.\.\]).*$\n?'
# `flags=re.MULTILINE` is necessary to match the start and end of each line
pattern = re.compile(regex, flags=re.MULTILINE)


def filter_lines(text):
    # replace the matched text with an empty string
    return pattern.sub('', text)


def on_env(env, config, files, **kwargs):
    env.filters['filter_lines'] = filter_lines
    return env
