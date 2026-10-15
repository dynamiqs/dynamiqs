// Set the copyright end year at view time, so that the footer of every documentation
// version stays up to date without being rebuilt.
document$.subscribe(() => {
    for (const year of document.querySelectorAll(".copyright-year")) {
        year.textContent = new Date().getFullYear();
    }
});
