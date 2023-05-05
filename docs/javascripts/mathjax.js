// code copied from https://squidfunk.github.io/mkdocs-material/setup/extensions/python-markdown-extensions/#arithmatex
window.MathJax = {
tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
},
options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
}
};

document$.subscribe(() => {
MathJax.typesetPromise()
})
