/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        "primary": "#2b6cee",
        "primary-dark": "#1a4bb0",
        "background-light": "#f6f6f8",
        "background-dark": "#101622",
        "surface-dark": "#18202F",
        "border-dark": "#2a3649",
        "success": "#22c55e",
        "error": "#ef4444",
        "warning": "#f59e0b",
      },
      fontFamily: {
        "display": ["Inter", "sans-serif"],
        "mono": ["JetBrains Mono", "monospace"],
      },
    },
  },
  plugins: [],
}
