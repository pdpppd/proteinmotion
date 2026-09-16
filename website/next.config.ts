import type { NextConfig } from "next";
const basePath = process.env.PAGES_BASE_PATH ?? "/proteinmotion";
const config: NextConfig = {
  output: "export",
  trailingSlash: true,
  basePath,
  images: { unoptimized: true },
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
};
export default config;
