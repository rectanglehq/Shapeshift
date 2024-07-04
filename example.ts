import { Shapeshift } from ".";


// Example usage
async function main() {
    const sourceObj = {
      personalInfo: {
        name: "John Doe",
        age: 30,
      },
      occupation: "Software Engineer",
      FullAddress: "123 Main St, Anytown",
      address: {
        street: "123 Main St",
        city: "Anytown"
      }
    };
  
    const targetObj = {
      fullName: "",
      yearsOld: 0,
      profession: "",
      location: {
        streetAddress: "",
        cityName: ""
      }
    };
  
    try {
        // Cohere example
      const shapeshifter = new Shapeshift(
        { embeddingClient: 'cohere', apiKey: process.env.COHERE_API_KEY || '' },
        { embeddingModel: 'embed-english-v3.0', similarityThreshold: 0.5 }
      );
      const shiftedObj = await shapeshifter.shapeshift(sourceObj, targetObj);
      console.log("Shifted object:", shiftedObj);
  
      // OpenAI example
      const openaiShapeshifter = new Shapeshift(
        { embeddingClient: 'openai', apiKey: process.env.OPENAI_API_KEY || '' },
        { embeddingModel: 'text-embedding-ada-002', similarityThreshold: 0.5 }
      );
      const openaiShiftedObj = await openaiShapeshifter.shapeshift(sourceObj, targetObj);
      console.log("OpenAI Shifted object:", openaiShiftedObj);
  
      // Voyage example
      const voyageShapeshifter = new Shapeshift(
        { embeddingClient: 'voyage', apiKey: process.env.VOYAGE_API_KEY || '' },
        { embeddingModel: 'voyage-large-2', similarityThreshold: 0.5 }
      );
      const voyageShiftedObj = await voyageShapeshifter.shapeshift(sourceObj, targetObj);
      console.log("Voyage Shifted object:", voyageShiftedObj);
    } catch (error) {
      console.error("Error:", error);
    }
  }
  
  main();