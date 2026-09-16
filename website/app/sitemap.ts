import type { MetadataRoute } from "next";
import { guides, site } from "@/lib/config";
export const dynamic = "force-static";
export default function sitemap(): MetadataRoute.Sitemap {
  return [
    { url: site + "/" },
    { url: site + "/gallery/" },
    ...guides.map((g) => ({ url: `${site}/docs/${g.slug}/` })),
  ];
}
