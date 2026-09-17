import type { MetadataRoute } from "next";
import { guides, site } from "@/lib/config";
import { referencePaths } from "@/lib/reference";
export const dynamic = "force-static";
export default function sitemap(): MetadataRoute.Sitemap {
  return [
    { url: site + "/" },
    { url: site + "/gallery/" },
    ...referencePaths.map((path) => ({
      url: `${site}/reference/${path.length ? path.join("/") + "/" : ""}`,
    })),
    ...guides.map((g) => ({ url: `${site}/docs/${g.slug}/` })),
  ];
}
