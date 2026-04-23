(function () {
  const TARGET_SRC_PREFIX = "/logo?theme=";
  const NEW_DARK_SRC = "/public/mediquery-chainlit-logo.svg";
  const NEW_LIGHT_SRC = "/public/mediquery-chainlit-logo-light.svg";
  const AVATAR_SRC_FRAGMENT = "/avatars/Assistant";
  const NEW_AVATAR_SRC = "/public/mediquery-icon.svg";

  function isDarkTheme() {
    const html = document.documentElement;
    const body = document.body;
    if (html.classList.contains("dark") || body.classList.contains("dark")) return true;

    const bg = getComputedStyle(body).backgroundColor || "";
    const m = bg.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/i);
    if (!m) return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    const r = Number(m[1]);
    const g = Number(m[2]);
    const b = Number(m[3]);
    const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    return luminance < 128;
  }

  function replaceLogo() {
    const nodes = document.querySelectorAll("img");
    const logoSrc = isDarkTheme() ? NEW_DARK_SRC : NEW_LIGHT_SRC;
    nodes.forEach((img) => {
      const src = img.getAttribute("src") || "";
      const alt = (img.getAttribute("alt") || "").toLowerCase();

      if (
        img.classList.contains("logo") &&
        (
          src.includes(TARGET_SRC_PREFIX) ||
          src.endsWith("/logo") ||
          src.includes("/public/mediquery-chainlit-logo.svg") ||
          src.includes("/public/mediquery-chainlit-logo-light.svg")
        )
      ) {
        img.setAttribute("src", logoSrc);
        img.setAttribute("alt", "MediQuery logo");
      }

      if (src.includes(AVATAR_SRC_FRAGMENT) || alt.includes("avatar for assistant")) {
        img.setAttribute("src", NEW_AVATAR_SRC);
        img.setAttribute("alt", "Avatar for Assistant");
      }
    });
  }

  replaceLogo();
  const observer = new MutationObserver(replaceLogo);
  observer.observe(document.documentElement, { childList: true, subtree: true });
  setInterval(replaceLogo, 1200);
})();
