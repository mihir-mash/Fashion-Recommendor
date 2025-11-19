import { useParams } from "react-router-dom";
import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Cloud, CloudRain, Sun, CloudDrizzle, Snowflake, Loader2 } from "lucide-react";
import Footer from "@/components/Footer";
import { useToast } from "@/hooks/use-toast";

const API_BASE_URL = "http://localhost:8000";

const iconMap: Record<string, any> = {
  humid: CloudDrizzle,
  cold: Snowflake,
  sunny: Sun,
  rainy: CloudRain,
  warm: Sun,
  mild: Cloud,
};

const CityWeather = () => {
  const { city } = useParams<{ city: string }>();
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();

  useEffect(() => {
    if (!city) {
      setError("City not specified");
      setLoading(false);
      return;
    }

    const fetchWeatherData = async () => {
      try {
        setLoading(true);
        const response = await fetch(
          `${API_BASE_URL}/recommend/weather?location=${encodeURIComponent(city)}&k=6`
        );
        
        if (!response.ok) {
          throw new Error(`Failed to fetch weather data: ${response.statusText}`);
        }
        
        const result = await response.json();
        setData(result);
        setError(null);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Failed to load weather data";
        setError(errorMessage);
        toast({
          title: "Error",
          description: errorMessage,
          variant: "destructive",
        });
      } finally {
        setLoading(false);
      }
    };

    fetchWeatherData();
  }, [city, toast]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-xl text-muted-foreground">
          {error || "City not found"}
        </p>
      </div>
    );
  }

  const weatherCondition = data.season?.toLowerCase() || "sunny";
  const WeatherIcon = iconMap[weatherCondition] || Sun;
  const temp = data.temp ? `${Math.round(data.temp)}°C` : "N/A";

  return (
    <div className="min-h-screen flex flex-col">
      <main className="flex-1">
        {/* Weather Section */}
        <section className="py-16 px-4 bg-gradient-to-b from-primary/5 to-background">
          <div className="container mx-auto text-center max-w-3xl">
            <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-6">
              {data.location || city}
            </h1>
            <WeatherIcon className="w-20 h-20 mx-auto mb-6 text-primary" />
            <p className="text-5xl font-bold text-primary mb-4">{temp}</p>
            <span className="inline-block px-4 py-2 bg-primary/10 text-primary rounded-full text-lg capitalize">
              {data.season || "N/A"}
            </span>
          </div>
        </section>

        {/* Outfit Recommendations */}
        <section className="py-12 px-4">
          <div className="container mx-auto max-w-6xl">
            <h2 className="text-3xl font-bold text-foreground mb-8 text-center">
              Recommended Outfits
            </h2>
            {data.results && data.results.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                {data.results.map((item: any, index: number) => (
                  <Card
                    key={item.id || index}
                    className="overflow-hidden transition-all hover:shadow-lg hover:scale-105"
                  >
                    <div className="aspect-[3/4] overflow-hidden bg-muted">
                      {item.image_url ? (
                        <img
                          src={`${API_BASE_URL}${item.image_url}`}
                          alt={item.product_display_name || "Outfit"}
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            (e.target as HTMLImageElement).src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='600'%3E%3Crect fill='%23ddd' width='400' height='600'/%3E%3Ctext x='50%25' y='50%25' text-anchor='middle' dy='.3em' fill='%23999'%3ENo Image%3C/text%3E%3C/svg%3E";
                          }}
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center text-muted-foreground">
                          No Image
                        </div>
                      )}
                    </div>
                    <CardContent className="p-4">
                      <h3 className="text-lg font-semibold text-foreground mb-2">
                        {item.product_display_name || `Item ${item.id}`}
                      </h3>
                      {item.score !== undefined && (
                        <span className="inline-block px-3 py-1 bg-primary/10 text-primary rounded-full text-sm">
                          Score: {item.score.toFixed(2)}
                        </span>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <p className="text-center text-muted-foreground">
                No recommendations available for this location.
              </p>
            )}
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default CityWeather;
