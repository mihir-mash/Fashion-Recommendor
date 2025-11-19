import { useNavigate } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { Cloud, CloudRain, Sun, CloudDrizzle, Snowflake } from "lucide-react";
import Footer from "@/components/Footer";

const cities = [
  { name: "Hyderabad", temp: "32°C", weather: "Humid", icon: CloudDrizzle },
  { name: "Shimla", temp: "15°C", weather: "Cold", icon: Snowflake },
  { name: "Goa", temp: "30°C", weather: "Sunny", icon: Sun },
  { name: "Shillong", temp: "20°C", weather: "Rainy", icon: CloudRain },
  { name: "Delhi", temp: "28°C", weather: "Sunny", icon: Sun },
];

const LocationSelector = () => {
  const navigate = useNavigate();

  const handleCityClick = (cityName: string) => {
    navigate(`/location/${cityName.toLowerCase()}`);
  };

  return (
    <div className="min-h-screen flex flex-col">
      <main className="flex-1">
        {/* Hero Section */}
        <section className="py-16 px-4 bg-gradient-to-b from-primary/5 to-background">
          <div className="container mx-auto text-center max-w-3xl">
            <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-4">
              Select Your City
            </h1>
            <p className="text-lg text-muted-foreground">
              Get outfit recommendations based on today's weather.
            </p>
          </div>
        </section>

        {/* Cities Grid */}
        <section className="py-12 px-4">
          <div className="container mx-auto max-w-5xl">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {cities.map((city) => {
                const WeatherIcon = city.icon;
                return (
                  <Card
                    key={city.name}
                    className="cursor-pointer transition-all hover:shadow-lg hover:scale-105"
                    onClick={() => handleCityClick(city.name)}
                  >
                    <CardContent className="p-6 text-center">
                      <WeatherIcon className="w-12 h-12 mx-auto mb-4 text-primary" />
                      <h3 className="text-xl font-semibold text-foreground mb-2">
                        {city.name}
                      </h3>
                      <p className="text-3xl font-bold text-primary mb-2">
                        {city.temp}
                      </p>
                      <span className="inline-block px-3 py-1 bg-primary/10 text-primary rounded-full text-sm">
                        {city.weather}
                      </span>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default LocationSelector;
