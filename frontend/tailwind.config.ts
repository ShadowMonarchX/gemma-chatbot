import type { Config } from "tailwindcss";

export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        heading: ["Space Grotesk", "ui-sans-serif", "sans-serif"],
        body: ["IBM Plex Sans", "ui-sans-serif", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"]
      },
      boxShadow: {
        panel: "0 12px 40px rgba(12, 20, 35, 0.12)",
        panelDark: "0 16px 42px rgba(0, 0, 0, 0.38)"
      },
      keyframes: {
        pulseDot: {
          "0%, 80%, 100%": { transform: "scale(0.85)", opacity: "0.45" },
          "40%": { transform: "scale(1)", opacity: "1" }
        },
        fadeUp: {
          from: { transform: "translateY(8px)", opacity: "0" },
          to: { transform: "translateY(0)", opacity: "1" }
        }
      },
      animation: {
        pulseDot: "pulseDot 1.2s infinite ease-in-out",
        fadeUp: "fadeUp 320ms ease-out"
      }
    }
  },
  plugins: []
} satisfies Config;
