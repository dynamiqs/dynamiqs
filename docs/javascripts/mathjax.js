// code copied from https://squidfunk.github.io/mkdocs-material/setup/extensions/python-markdown-extensions/#arithmatex
window.MathJax = {
    loader: { load: ['[tex]/braket'] },  // <-- added
    tex: {
        inlineMath: [["\\(", "\\)"]],
        displayMath: [["\\[", "\\]"]],
        processEscapes: true,
        processEnvironments: true,
        packages: { '[+]': ['braket'] },  // <-- added
        macros: {  // <-- added
            dag: '\\dagger',
            dd: '\\mathrm{d}',
            dt: '\\mathrm{d}t',
            tr: ["\\mathrm{Tr}\\left[#1\\right]", 1]
        },
    },
    options: {
        ignoreHtmlClass: ".*|",
        processHtmlClass: "arithmatex"
    }
};

document$.subscribe(() => {
    MathJax.typesetPromise()
})
