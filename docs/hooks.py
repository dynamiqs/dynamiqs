import re


def replace_render(text):
    # This regex will match the word 'render' followed by anything inside parentheses
    regex = (
        r"(<span class=\"gp\">&gt;&gt;&gt; </span>)?<span"
        r" class=\"n\">render<\/span><span class=\"p\">\(<\/span><span"
        r" class=\"s1\">[^']*<\/span><span class=\"p\">\)<\/span>\n"
    )
    pattern = re.compile(regex)
    # Replace the matched text with an empty string (or whatever you want)
    return pattern.sub('', text)


def on_env(env, config, files, **kwargs):
    env.filters['replace_render'] = replace_render
    return env
