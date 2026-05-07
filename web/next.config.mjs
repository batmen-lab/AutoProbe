/** @type {import('next').NextConfig} */
// The frontend calls the FastAPI backend directly (CORS-allowed) instead of
// going through a Next.js rewrite — rewrites time out around 30s, but our NLP
// chains routinely run longer. Configure the URL with NEXT_PUBLIC_API_BASE.
const nextConfig = {};

export default nextConfig;
