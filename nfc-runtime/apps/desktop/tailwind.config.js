/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0b0f14",
          900: "#121820",
          800: "#1a2330",
          700: "#243044",
          600: "#3a4d66",
        },
        signal: {
          DEFAULT: "#3ecf8e",
          dim: "#2a9d6a",
          muted: "#1a5c42",
        },
        amber: {
          soft: "#d4a574",
        },
        mist: {
          DEFAULT: "#c8d0dc",
          dim: "#8b97a8",
        },
      },
      fontFamily: {
        display: ['"IBM Plex Sans"', "system-ui", "sans-serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "monospace"],
      },
      boxShadow: {
        panel: "0 0 0 1px rgba(62, 207, 142, 0.08), 0 12px 40px rgba(0,0,0,0.35)",
      },
    },
  },
  plugins: [],
};
