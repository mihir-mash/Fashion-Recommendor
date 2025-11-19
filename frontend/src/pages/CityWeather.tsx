import { useParams } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { Cloud, CloudRain, Sun, CloudDrizzle, Snowflake } from "lucide-react";
import Footer from "@/components/Footer";
import outfit1 from "@/assets/outfit-1.jpg";
import outfit2 from "@/assets/outfit-2.jpg";
import outfit3 from "@/assets/outfit-3.jpg";
import outfit4 from "@/assets/outfit-4.jpg";

const cityData: Record<string, any> = {
  hyderabad: {
    name: "Hyderabad",
    temp: "32°C",
    weather: "Humid",
    icon: CloudDrizzle,
    outfits: [
      { image: outfit1, name: "Cotton Kurta Set", tag: "Best for humid weather" },
      { image: outfit2, name: "Light Linen Shirt", tag: "Perfect for hot days" },
      { image: outfit3, name: "Breathable Dress", tag: "Ideal for summer" },
      { image: outfit4, name: "Casual Tee & Shorts", tag: "Stay cool & comfortable" },
    ],
  },
  shimla: {
    name: "Shimla",
    temp: "15°C",
    weather: "Cold",
    icon: Snowflake,
    outfits: [
      { image: outfit1, name: "Woolen Sweater", tag: "Perfect for cold weather" },
      { image: outfit2, name: "Winter Jacket", tag: "Stay warm & stylish" },
      { image: outfit3, name: "Layered Outfit", tag: "Great for chilly days" },
      { image: outfit4, name: "Cozy Cardigan", tag: "Best for mountain cold" },
    ],
  },
  goa: {
    name: "Goa",
    temp: "30°C",
    weather: "Sunny",
    icon: Sun,
    outfits: [
      { image: outfit1, name: "Beach Dress", tag: "Perfect for sunny days" },
      { image: outfit2, name: "Casual Shorts Set", tag: "Ideal for beach vibes" },
      { image: outfit3, name: "Flowy Sundress", tag: "Best for coastal heat" },
      { image: outfit4, name: "Tank & Shorts", tag: "Stay breezy" },
    ],
  },
  shillong: {
    name: "Shillong",
    temp: "20°C",
    weather: "Rainy",
    icon: CloudRain,
    outfits: [
      { image: outfit1, name: "Waterproof Jacket", tag: "Great for monsoon" },
      { image: outfit2, name: "Rain-Ready Attire", tag: "Perfect for rainy days" },
      { image: outfit3, name: "Light Raincoat Look", tag: "Stay dry & stylish" },
      { image: outfit4, name: "Monsoon Casual", tag: "Best for wet weather" },
    ],
  },
  delhi: {
    name: "Delhi",
    temp: "28°C",
    weather: "Sunny",
    icon: Sun,
    outfits: [
      { image: outfit1, name: "Smart Casuals", tag: "Perfect for sunny days" },
      { image: outfit2, name: "Trendy Streetwear", tag: "Ideal for city heat" },
      { image: outfit3, name: "Summer Formal", tag: "Best for warm weather" },
      { image: outfit4, name: "Breezy Outfit", tag: "Stay comfortable" },
    ],
  },
};

const CityWeather = () => {
  const { city } = useParams<{ city: string }>();
  const data = cityData[city?.toLowerCase() || ""];

  if (!data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-xl text-muted-foreground">City not found</p>
      </div>
    );
  }

  const WeatherIcon = data.icon;

  return (
    <div className="min-h-screen flex flex-col">
      <main className="flex-1">
        {/* Weather Section */}
        <section className="py-16 px-4 bg-gradient-to-b from-primary/5 to-background">
          <div className="container mx-auto text-center max-w-3xl">
            <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-6">
              {data.name}
            </h1>
            <WeatherIcon className="w-20 h-20 mx-auto mb-6 text-primary" />
            <p className="text-5xl font-bold text-primary mb-4">{data.temp}</p>
            <span className="inline-block px-4 py-2 bg-primary/10 text-primary rounded-full text-lg">
              {data.weather}
            </span>
          </div>
        </section>

        {/* Outfit Recommendations */}
        <section className="py-12 px-4">
          <div className="container mx-auto max-w-6xl">
            <h2 className="text-3xl font-bold text-foreground mb-8 text-center">
              Recommended Outfits
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              {data.outfits.map((outfit: any, index: number) => (
                <Card
                  key={index}
                  className="overflow-hidden transition-all hover:shadow-lg hover:scale-105"
                >
                  <div className="aspect-[3/4] overflow-hidden">
                    <img
                      src={outfit.image}
                      alt={outfit.name}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <CardContent className="p-4">
                    <h3 className="text-lg font-semibold text-foreground mb-2">
                      {outfit.name}
                    </h3>
                    <span className="inline-block px-3 py-1 bg-primary/10 text-primary rounded-full text-sm">
                      {outfit.tag}
                    </span>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default CityWeather;
