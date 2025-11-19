import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Upload } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import Footer from "@/components/Footer";

const OutfitMatch = () => {
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>("");
  const [isMatching, setIsMatching] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const { toast } = useToast();

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleMatch = () => {
    if (!image) {
      toast({
        title: "Image Required",
        description: "Please upload an outfit image to match.",
        variant: "destructive",
      });
      return;
    }

    setIsMatching(true);

    // Simulate matching with dummy data
    setTimeout(() => {
      const dummyResults = [
        {
          id: 1,
          category: "Accessories",
          name: "Leather Belt",
          note: "Complements your outfit perfectly",
          image: "https://images.unsplash.com/photo-1624222247344-550fb60583dc?w=500",
        },
        {
          id: 2,
          category: "Shoes",
          name: "Canvas Sneakers",
          note: "Casual and comfortable match",
          image: "https://images.unsplash.com/photo-1549298916-b41d501d3772?w=500",
        },
        {
          id: 3,
          category: "Jacket",
          name: "Bomber Jacket",
          note: "Perfect layering piece",
          image: "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=500",
        },
        {
          id: 4,
          category: "Similar Items",
          name: "Alternative Shirt",
          note: "Similar style, different color",
          image: "https://images.unsplash.com/photo-1620799140188-3b2a02fd9a77?w=500",
        },
      ];

      setResults(dummyResults);
      setIsMatching(false);
      toast({
        title: "Match Complete!",
        description: "Found items that complement your outfit.",
      });
    }, 2000);
  };

  return (
    <div className="min-h-screen">
      <div className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl font-bold text-foreground mb-4">
            Match Your Outfit
          </h1>
          <p className="text-muted-foreground mb-8">
            Upload an image and we'll find items that complement your look
          </p>

          <Card className="p-6 mb-8">
            <div className="space-y-6">
              {/* Image Upload */}
              <div>
                <label
                  htmlFor="outfit-upload"
                  className="flex flex-col items-center justify-center w-full min-h-[300px] border-2 border-dashed border-border rounded-lg cursor-pointer hover:border-primary transition-colors"
                >
                  {imagePreview ? (
                    <img
                      src={imagePreview}
                      alt="Uploaded outfit"
                      className="max-h-[400px] object-contain rounded-lg"
                    />
                  ) : (
                    <div className="text-center p-8">
                      <Upload className="mx-auto h-16 w-16 text-muted-foreground mb-4" />
                      <p className="text-lg font-medium text-foreground mb-2">
                        Upload Your Outfit
                      </p>
                      <p className="text-sm text-muted-foreground">
                        Drag and drop or click to browse
                      </p>
                    </div>
                  )}
                  <input
                    id="outfit-upload"
                    type="file"
                    className="hidden"
                    accept="image/*"
                    onChange={handleImageChange}
                  />
                </label>
              </div>

              <Button
                onClick={handleMatch}
                disabled={isMatching}
                className="w-full"
                size="lg"
              >
                {isMatching ? "Matching..." : "Match Outfit"}
              </Button>
            </div>
          </Card>

          {/* Results */}
          {results.length > 0 && (
            <div className="animate-fade-in">
              <h2 className="text-2xl font-bold text-foreground mb-6">
                Complementary Items
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                {results.map((item) => (
                  <Card
                    key={item.id}
                    className="overflow-hidden border-0 shadow-[var(--shadow-card)] hover:shadow-[var(--shadow-card-hover)] transition-all duration-300"
                  >
                    <div className="aspect-square overflow-hidden">
                      <img
                        src={item.image}
                        alt={item.name}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div className="p-4">
                      <p className="text-xs text-primary font-medium mb-1">
                        {item.category}
                      </p>
                      <h3 className="font-semibold text-foreground mb-2">
                        {item.name}
                      </h3>
                      <p className="text-sm text-muted-foreground">{item.note}</p>
                    </div>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default OutfitMatch;
