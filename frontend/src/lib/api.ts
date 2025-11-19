/**
 * API service for connecting to the backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export interface SearchResult {
  id: string;
  product_display_name?: string;
  image_url?: string;
  score?: number;
  short_explain?: string;
}

export interface WeatherRecommendation {
  location: string;
  season: string;
  temp?: number;
  total_candidates: number;
  results: SearchResult[];
}

export interface MatchResult {
  id: string;
  score: number;
  product_display_name?: string;
  image_url?: string;
  short_explain?: string;
}

/**
 * Search for items by text query
 */
export async function searchByText(text: string, k: number = 6): Promise<SearchResult[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/search/text`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text, k }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorMsg = errorData.detail?.details || errorData.detail?.error || response.statusText;
      throw new Error(`Search failed: ${errorMsg}`);
    }

    const data = await response.json();
    return data.results || [];
  } catch (error) {
    console.error("Error searching by text:", error);
    throw error;
  }
}

/**
 * Search for items by image
 */
export async function searchByImage(imageFile: File, k: number = 6): Promise<SearchResult[]> {
  try {
    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("k", k.toString());

    const response = await fetch(`${API_BASE_URL}/search/image`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorMsg = errorData.detail?.details || errorData.detail?.error || response.statusText;
      throw new Error(`Image search failed: ${errorMsg}`);
    }

    const data = await response.json();
    return data.results || [];
  } catch (error) {
    console.error("Error searching by image:", error);
    throw error;
  }
}

/**
 * Get weather-based recommendations for a city
 */
export async function getWeatherRecommendations(
  location: string,
  k: number = 6
): Promise<WeatherRecommendation> {
  try {
    const response = await fetch(
      `${API_BASE_URL}/recommend/weather?location=${encodeURIComponent(location)}&k=${k}`
    );

    if (!response.ok) {
      throw new Error(`Weather recommendation failed: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error getting weather recommendations:", error);
    throw error;
  }
}

/**
 * Match outfit items from an uploaded image
 */
export async function matchOutfit(
  imageFile: File,
  targetType?: string,
  k: number = 6
): Promise<MatchResult[]> {
  try {
    const formData = new FormData();
    formData.append("image", imageFile);
    if (targetType) {
      formData.append("target_type", targetType);
    }
    formData.append("k", k.toString());

    const response = await fetch(`${API_BASE_URL}/match/outfit`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorMsg = errorData.detail?.details || errorData.detail?.error || response.statusText;
      throw new Error(`Outfit matching failed: ${errorMsg}`);
    }

    const data = await response.json();
    return data.results || [];
  } catch (error) {
    console.error("Error matching outfit:", error);
    throw error;
  }
}

/**
 * Get image URL (handles both relative and absolute URLs)
 */
export function getImageUrl(imageUrl?: string): string {
  if (!imageUrl) {
    return "";
  }
  
  // If it's already a full URL, return as is
  if (imageUrl.startsWith("http://") || imageUrl.startsWith("https://")) {
    return imageUrl;
  }
  
  // If it's a relative URL, prepend API base URL
  return `${API_BASE_URL}${imageUrl.startsWith("/") ? "" : "/"}${imageUrl}`;
}

