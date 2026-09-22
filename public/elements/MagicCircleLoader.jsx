// public/elements/MagicCircleLoader.jsx

export default function MagicCircleLoader() {
  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      padding: "24px",
      gap: "12px",
    }}>
      <svg
        width="140"
        height="140"
        viewBox="0 0 200 200"
        xmlns="http://www.w3.org/2000/svg"
        style={{ filter: "drop-shadow(0 0 12px rgba(212,175,55,0.6))" }}
      >
        <defs>
          <linearGradient id="goldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#ffb700" />
            <stop offset="100%" stopColor="#d4af37" />
          </linearGradient>
        </defs>

        {/* Outer ring — slow clockwise */}
        <g style={{ transformOrigin: "100px 100px", animation: "spin 8s linear infinite" }}>
          <circle cx="100" cy="100" r="92" fill="none" stroke="url(#goldGrad)" strokeWidth="1.5" />
          <circle cx="100" cy="100" r="88" fill="none" stroke="#d4af37" strokeWidth="0.5" strokeDasharray="2 6" />
        </g>

        {/* Middle ring — counter-clockwise */}
        <g style={{ transformOrigin: "100px 100px", animation: "spin 5s linear infinite reverse" }}>
          <circle cx="100" cy="100" r="72" fill="none" stroke="#ffb700" strokeWidth="1" />
          {[...Array(8)].map((_, i) => {
            const angle = (i / 8) * Math.PI * 2;
            const x = 100 + 72 * Math.cos(angle);
            const y = 100 + 72 * Math.sin(angle);
            return <circle key={i} cx={x} cy={y} r="2.5" fill="#ffb700" />;
          })}
        </g>

        {/* Inner rotated square — clockwise */}
        <g style={{ transformOrigin: "100px 100px", animation: "spin 6s linear infinite" }}>
          <rect
            x="65" y="65" width="70" height="70"
            fill="none" stroke="#d4af37" strokeWidth="1"
            transform="rotate(45 100 100)"
          />
        </g>

        {/* Inner rotating triangle — counter-clockwise */}
        <g style={{ transformOrigin: "100px 100px", animation: "spin 4s linear infinite reverse" }}>
          <polygon
            points="100,55 140,125 60,125"
            fill="none" stroke="#ffb700" strokeWidth="1"
          />
        </g>

        {/* Center sigil — pulsing */}
        <g style={{ transformOrigin: "100px 100px", animation: "pulse 1.6s ease-in-out infinite" }}>
          <circle cx="100" cy="100" r="8" fill="#ffb700" opacity="0.9" />
          <circle cx="100" cy="100" r="14" fill="none" stroke="#ffb700" strokeWidth="1" opacity="0.6" />
        </g>

        <style>{`
          @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
          @keyframes pulse { 0%,100% { opacity: 0.5; } 50% { opacity: 1; } }
        `}</style>
      </svg>

      <p style={{
        color: "#d4af37",
        fontFamily: "Georgia, serif",
        fontSize: "13px",
        letterSpacing: "2px",
        textTransform: "uppercase",
        margin: 0,
        animation: "pulse 1.6s ease-in-out infinite",
      }}>
        Analyzing
      </p>
    </div>
  );
}