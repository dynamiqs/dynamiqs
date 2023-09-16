document$.subscribe(({ body }) => {
    const macros = {
        "\\dag": '\\dagger',
        "\\dd": '\\mathrm{d}',
        "\\dt": '\\mathrm{d}t',
        "\\tr": "\\mathrm{Tr}\\left[#1\\right]",
    };

    renderMathInElement(body, {
        delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
            { left: "\\(", right: "\\)", display: false },
            { left: "\\[", right: "\\]", display: true }
        ],
        macros: macros,
    })
});
