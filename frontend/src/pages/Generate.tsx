import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Upload, Sparkles } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import Footer from "@/components/Footer";

const Generate = () => {
  const [description, setDescription] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>("");
  const [isGenerating, setIsGenerating] = useState(false);
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

  const handleGenerate = () => {
    if (!description.trim() && !image) {
      toast({
        title: "Input Required",
        description: "Please describe what you're looking for or upload an image.",
        variant: "destructive",
      });
      return;
    }

    setIsGenerating(true);

    // Simulate generation with dummy data
    setTimeout(() => {
      const dummyResults = [
        {
          id: 1,
          name: "Elegant Summer Dress",
          description: "Perfect for warm weather with breathable fabric",
          image: "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=500",
        },
        {
          id: 2,
          name: "Casual Denim Jacket",
          description: "Versatile layering piece for any occasion",
          image: "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=500",
        },
        {
          id: 3,
          name: "Classic White Sneakers",
          description: "Comfortable and stylish everyday footwear",
          image: "https://images.unsplash.com/photo-1549298916-b41d501d3772?w=500",
        },
        {
          id: 4,
          name: "Cozy Knit Sweater",
          description: "Warm and comfortable for cooler days",
          image: "https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=500",
        },
      ];

      setResults(dummyResults);
      setIsGenerating(false);
      toast({
        title: "Recommendations Ready!",
        description: "Here are some outfit suggestions for you.",
      });
    }, 2000);
  };

  return (
    <div className="min-h-screen">
      <div className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl font-bold text-foreground mb-4">
            Generate Outfit
          </h1>
          <p className="text-muted-foreground mb-8">
            Describe what you're looking for or upload a reference image
          </p>

          <Card className="p-6 mb-8">
            <div className="space-y-6">
              {/* Text Input */}
              <div>
                <Label htmlFor="description" className="text-base mb-2">
                  What are you looking for?
                </Label>
                <Textarea
                  id="description"
                  placeholder="e.g., party wear, winter jacket, streetwear fit..."
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="min-h-[120px] resize-none"
                />
              </div>

              {/* Image Upload */}
              <div>
                <Label className="text-base mb-2">Reference Image (Optional)</Label>
                <div className="mt-2">
                  <label
                    htmlFor="image-upload"
                    className="flex flex-col items-center justify-center w-full h-40 border-2 border-dashed border-border rounded-lg cursor-pointer hover:border-primary transition-colors"
                  >
                    {imagePreview ? (
                      <img
                        src={imagePreview}
                        alt="Preview"
                        className="h-full object-contain rounded-lg"
                      />
                    ) : (
                      <div className="text-center">
                        <Upload className="mx-auto h-10 w-10 text-muted-foreground mb-2" />
                        <p className="text-sm text-muted-foreground">
                          Click to upload or drag and drop
                        </p>
                      </div>
                    )}
                    <input
                      id="image-upload"
                      type="file"
                      className="hidden"
                      accept="image/*"
                      onChange={handleImageChange}
                    />
                  </label>
                </div>
              </div>

              <Button
                onClick={handleGenerate}
                disabled={isGenerating}
                className="w-full"
                size="lg"
              >
                {isGenerating ? (
                  <>
                    <Sparkles className="mr-2 h-4 w-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="mr-2 h-4 w-4" />
                    Generate
                  </>
                )}
              </Button>
            </div>
          </Card>

          {/* Results */}
          {results.length > 0 && (
            <div className="animate-fade-in">
              <h2 className="text-2xl font-bold text-foreground mb-6">
                Recommended for You
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
                      <h3 className="font-semibold text-foreground mb-2">
                        {item.name}
                      </h3>
                      <p className="text-sm text-muted-foreground">
                        {item.description}
                      </p>
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

export default Generate;
