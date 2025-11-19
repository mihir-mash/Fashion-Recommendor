import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useNavigate } from "react-router-dom";
import { Upload, Sparkles, ShoppingBag } from "lucide-react";
import Footer from "@/components/Footer";
import heroBanner from "@/assets/hero-banner.jpg";
import outfit1 from "@/assets/outfit-1.jpg";
import outfit2 from "@/assets/outfit-2.jpg";
import outfit3 from "@/assets/outfit-3.jpg";
import outfit4 from "@/assets/outfit-4.jpg";

const Home = () => {
  const navigate = useNavigate();

  const outfits = [
    {
      id: 1,
      image: outfit1,
      name: "Casual Comfort",
      weather: "Sunny",
    },
    {
      id: 2,
      image: outfit2,
      name: "Winter Warmth",
      weather: "Cold",
    },
    {
      id: 3,
      image: outfit3,
      name: "Summer Breeze",
      weather: "Sunny",
    },
    {
      id: 4,
      image: outfit4,
      name: "Rainy Day Ready",
      weather: "Rainy",
    },
  ];

  const steps = [
    {
      icon: Upload,
      title: "Upload or Select",
      description: "Choose your clothing preferences or upload an image",
    },
    {
      icon: Sparkles,
      title: "Style Analysis",
      description: "We analyze based on weather and your personal look",
    },
    {
      icon: ShoppingBag,
      title: "Get Recommendations",
      description: "Receive instant personalized outfit suggestions",
    },
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative h-[600px] overflow-hidden">
        <div
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${heroBanner})` }}
        >
          <div className="absolute inset-0 bg-gradient-to-r from-background/95 via-background/70 to-transparent" />
        </div>
        
        <div className="relative container mx-auto px-4 h-full flex items-center">
          <div className="max-w-2xl">
            <h1 className="text-5xl md:text-6xl font-bold text-foreground mb-6 leading-tight">
              Outfitly — Your Stylist
            </h1>
            <p className="text-xl text-muted-foreground mb-8">
              Get outfit suggestions or upload an image to match your look.
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <Button
                size="lg"
                onClick={() => navigate("/generate")}
                className="text-base"
              >
                Generate Outfit
              </Button>
              <Button
                size="lg"
                variant="outline"
                onClick={() => navigate("/match")}
                className="text-base"
              >
                Match My Outfit
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Weather-Based Recommendations */}
      <section className="container mx-auto px-4 py-20">
        <h2 className="text-3xl font-bold text-foreground mb-12 text-center">
          Outfits for Today
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {outfits.map((outfit) => (
            <Card
              key={outfit.id}
              className="overflow-hidden border-0 shadow-[var(--shadow-card)] hover:shadow-[var(--shadow-card-hover)] transition-all duration-300 hover:scale-[1.02] cursor-pointer"
            >
              <div className="aspect-square overflow-hidden">
                <img
                  src={outfit.image}
                  alt={outfit.name}
                  className="w-full h-full object-cover transition-transform duration-300 hover:scale-105"
                />
              </div>
              <div className="p-4">
                <h3 className="font-semibold text-foreground mb-2">{outfit.name}</h3>
                <Badge variant="secondary" className="text-xs">
                  {outfit.weather}
                </Badge>
              </div>
            </Card>
          ))}
        </div>
      </section>

      {/* How It Works */}
      <section className="bg-muted py-20">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold text-foreground mb-12 text-center">
            How It Works
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            {steps.map((step, index) => (
              <div key={index} className="text-center">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary text-primary-foreground mb-6">
                  <step.icon size={28} />
                </div>
                <h3 className="text-xl font-semibold text-foreground mb-3">
                  {step.title}
                </h3>
                <p className="text-muted-foreground">{step.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Home;
