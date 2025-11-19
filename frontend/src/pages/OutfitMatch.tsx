import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Upload } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import Footer from "@/components/Footer";
import { matchOutfit, getImageUrl } from "@/lib/api";

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

  const handleMatch = async () => {
    if (!image) {
      toast({
        title: "Image Required",
        description: "Please upload an outfit image to match.",
        variant: "destructive",
      });
      return;
    }

    setIsMatching(true);
    setResults([]);

    try {
      const matchResults = await matchOutfit(image, undefined, 6);

      // Transform results to match component expectations
      const transformedResults = matchResults.map((item) => ({
        id: item.id,
        category: "Fashion Item",
        name: item.product_display_name || `Item ${item.id}`,
        note: item.short_explain || "Complements your outfit",
        image: getImageUrl(item.image_url),
        score: item.score,
      }));

      setResults(transformedResults);
      toast({
        title: "Match Complete!",
        description: `Found ${transformedResults.length} items that complement your outfit.`,
      });
    } catch (error) {
      console.error("Error matching outfit:", error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to match outfit. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsMatching(false);
    }
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
                    <div className="aspect-square overflow-hidden bg-muted">
                      {item.image ? (
                        <img
                          src={item.image}
                          alt={item.name}
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            (e.target as HTMLImageElement).src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Crect fill='%23ddd' width='400' height='400'/%3E%3Ctext x='50%25' y='50%25' text-anchor='middle' dy='.3em' fill='%23999'%3ENo Image%3C/text%3E%3C/svg%3E";
                          }}
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center text-muted-foreground">
                          No Image
                        </div>
                      )}
                    </div>
                    <div className="p-4">
                      <p className="text-xs text-primary font-medium mb-1">
                        {item.category}
                      </p>
                      <h3 className="font-semibold text-foreground mb-2">
                        {item.name}
                      </h3>
                      <p className="text-sm text-muted-foreground mb-2">{item.note}</p>
                      {item.score !== undefined && (
                        <span className="inline-block px-2 py-1 bg-primary/10 text-primary rounded text-xs">
                          Score: {item.score.toFixed(2)}
                        </span>
                      )}
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
