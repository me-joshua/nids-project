/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'chat-bg': '#1a1a1a',
        'chat-sidebar': '#0d1117',
        'chat-input': '#262626',
        'chat-text': '#e6e6e6',
        'chat-border': '#404040',
        'threat-high': '#ff4757',
        'threat-medium': '#ffa502',
        'threat-low': '#26de81',
        'threat-normal': '#74b9ff'
      },
      animation: {
        'pulse-slow': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'typing': 'typing 1.5s infinite'
      }
    },
  },
  plugins: [],
}
