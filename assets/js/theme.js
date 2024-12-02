const STORAGE_KEY = "theme";
const THEME_ATTR  = "data-theme";
const QUERY_KEY   = "(prefers-color-scheme: light)";

const themes = {
  LIGHT: "light",
};

initTheme();


function initTheme() {
    setTheme(themes.LIGHT);
}
  
function toggleTheme() {
    setTheme(themes.LIGHT);
    localStorage.setItem(STORAGE_KEY, themes.LIGHT);
}
  
function setTheme(value) {
    document.documentElement.setAttribute(THEME_ATTR, themes.LIGHT);
}