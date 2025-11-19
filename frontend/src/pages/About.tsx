import { Card } from "@/components/ui/card";
import { Sparkles, Users, Heart } from "lucide-react";
import Footer from "@/components/Footer";

const About = () => {
  const values = [
    {
      icon: Sparkles,
      title: "Style Made Simple",
      description:
        "We believe fashion should be accessible and effortless for everyone.",
    },
    {
      icon: Users,
      title: "Personalized for You",
      description:
        "Every recommendation is tailored to your unique style and preferences.",
    },
    {
      icon: Heart,
      title: "Confidence First",
      description:
        "Our mission is to help you feel confident in what you wear, every day.",
    },
  ];

  return (
    <div className="min-h-screen">
      <div className="container mx-auto px-4 py-12">
        {/* Hero Section */}
        <div className="max-w-3xl mx-auto text-center mb-16">
          <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-6">
            About Outfitly
          </h1>
          <p className="text-lg text-muted-foreground">
            Your personal fashion companion, helping you discover the perfect
            outfit for any occasion. We combine style expertise with modern
            technology to make fashion accessible and enjoyable for everyone.
          </p>
        </div>

        {/* Values Section */}
        <div className="max-w-5xl mx-auto mb-16">
          <h2 className="text-3xl font-bold text-foreground mb-8 text-center">
            What We Stand For
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {values.map((value, index) => (
              <Card
                key={index}
                className="p-6 text-center border-0 shadow-[var(--shadow-card)] hover:shadow-[var(--shadow-card-hover)] transition-all duration-300"
              >
                <div className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-primary text-primary-foreground mb-4">
                  <value.icon size={24} />
                </div>
                <h3 className="text-xl font-semibold text-foreground mb-3">
                  {value.title}
                </h3>
                <p className="text-muted-foreground">{value.description}</p>
              </Card>
            ))}
          </div>
        </div>

        {/* Story Section */}
        <div className="max-w-3xl mx-auto">
          <Card className="p-8 border-0 shadow-[var(--shadow-card)]">
            <h2 className="text-2xl font-bold text-foreground mb-6">
              Our Story
            </h2>
            <div className="space-y-4 text-muted-foreground">
              <p>
                Outfitly was born from a simple idea: everyone deserves to feel
                confident in what they wear, without the stress of endless
                scrolling and decision fatigue.
              </p>
              <p>
                We understand that finding the right outfit can be
                overwhelming. That's why we created a platform that combines
                personalized recommendations with weather-aware suggestions,
                making it easier than ever to look and feel your best.
              </p>
              <p>
                Whether you're getting ready for a special occasion or just
                planning your daily look, Outfitly is here to guide you every
                step of the way.
              </p>
            </div>
          </Card>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default About;
