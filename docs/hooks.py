import re


def replace_render(text):
    # This regex will match the word 'render' followed by anything inside parentheses
    regex1 = (
        r"(<span class=\"gp\">&gt;&gt;&gt; </span>)?<span"
        r" class=\"n\">render<\/span><span class=\"p\">\(<\/span><span"
        r" class=\"s1\">[^']*<\/span><span class=\"p\">\)<\/span>\n"
    )
    regex2 = r"<p>% skip: start</p>\n"
    regex3 = r"<p>% skip: end</p>\n"
    pattern = re.compile(f"{regex1}|{regex2}|{regex3}")
    # Replace the matched text with an empty string (or whatever you want)
    return pattern.sub('', text)


def on_env(env, config, files, **kwargs):
    env.filters['replace_render'] = replace_render
    return env
