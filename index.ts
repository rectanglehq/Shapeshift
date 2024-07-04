// Library to map arbitrarily strucutred JSON objects using vector embeddings 
import { CohereClient } from 'cohere-ai';
import OpenAI from "openai";
import axios from 'axios';

// Define types
type Embedding = number[];
type ObjectWithStringKeys = { [key: string]: any };

type ShapeshiftOptions = {
  embeddingModel?: string;
  similarityThreshold?: number;
};

type EmbeddingClient = 'cohere' | 'openai' | 'voyage';

class Shapeshift {
  private cohere: CohereClient | null;
  private openai: OpenAI | null;
  private voyageApiKey: string | null;
  private embeddingClient: EmbeddingClient;
  private embeddingModel: string;
  private similarityThreshold: number;

  constructor({ embeddingClient, apiKey }: { embeddingClient: EmbeddingClient, apiKey: string }, options: ShapeshiftOptions = {}) {
    this.embeddingClient = embeddingClient;
    if (embeddingClient === 'cohere') {
      this.cohere = new CohereClient({ token: apiKey });
      this.openai = null;
      this.voyageApiKey = null;
      this.embeddingModel = options.embeddingModel || 'embed-english-v3.0';
    } else if (embeddingClient === 'openai') {
      this.openai = new OpenAI({ apiKey });
      this.cohere = null;
      this.voyageApiKey = null;
      this.embeddingModel = options.embeddingModel || 'text-embedding-ada-002';
    } else if (embeddingClient === 'voyage') {
      this.voyageApiKey = apiKey;
      this.cohere = null;
      this.openai = null;
      this.embeddingModel = options.embeddingModel || 'voyage-large-2';
    } else {
      throw new Error('Unsupported embedding client');
    }
    this.similarityThreshold = options.similarityThreshold || 0.5; // Default threshold of 0.5
  }

  private async calculateEmbeddings(texts: string[]): Promise<Embedding[]> {
    if (this.embeddingClient === 'cohere' && this.cohere) {
      const response = await this.cohere.embed({
        texts: texts,
        model: this.embeddingModel,
        inputType: 'classification',
      });
      return response.embeddings as Embedding[];
    } else if (this.embeddingClient === 'openai' && this.openai) {
      const embeddings = await Promise.all(texts.map(async (text) => {
        const response = await this.openai!.embeddings.create({
          model: this.embeddingModel,
          input: text,
          encoding_format: "float",
        });
        return response.data[0].embedding;
      }));
      return embeddings;
    } else if (this.embeddingClient === 'voyage' && this.voyageApiKey) {
      
      const response = await axios.post(
        'https://api.voyageai.com/v1/embeddings',
        {
          input: texts,
          model: this.embeddingModel,
        },
        {
          headers: {
            'Authorization': `Bearer ${this.voyageApiKey}`,
            'Content-Type': 'application/json',
          },
        }
      );
      return response.data.data.map((item: any) => item.embedding);
    } else {
      throw new Error('Embedding client not properly initialized');
    }
  }

  private cosineSimilarity(vecA: Embedding, vecB: Embedding): number {
    const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
    return dotProduct / (magnitudeA * magnitudeB);
  }

  private findClosestMatch(sourceEmbedding: Embedding, targetEmbeddings: Embedding[]): number | null {
    let maxSimilarity = -Infinity;
    let closestIndex = -1;

    for (let i = 0; i < targetEmbeddings.length; i++) {
      const similarity = this.cosineSimilarity(sourceEmbedding, targetEmbeddings[i]);
      if (similarity > maxSimilarity) {
        maxSimilarity = similarity;
        closestIndex = i;
      }
    }

    return maxSimilarity >= this.similarityThreshold ? closestIndex : null;
  }

  private flattenObject(obj: ObjectWithStringKeys, prefix = ''): ObjectWithStringKeys {
    return Object.keys(obj).reduce((acc: ObjectWithStringKeys, k) => {
      const pre = prefix.length ? prefix + '.' : '';
      if (typeof obj[k] === 'object' && obj[k] !== null && !Array.isArray(obj[k])) {
        Object.assign(acc, this.flattenObject(obj[k], pre + k));
      } else {
        acc[pre + k] = obj[k];
      }
      return acc;
    }, {});
  }

  private unflattenObject(obj: ObjectWithStringKeys): ObjectWithStringKeys {
    const result: ObjectWithStringKeys = {};
    for (const key in obj) {
      const keys = key.split('.');
      keys.reduce((r: ObjectWithStringKeys, k: string, i: number) => {
        return r[k] = i === keys.length - 1 ? obj[key] : (r[k] || {});
      }, result);
    }
    return result;
  }

  async shapeshift<T extends ObjectWithStringKeys, U extends ObjectWithStringKeys>(
    sourceObj: T,
    targetObj: U
  ): Promise<U> {
    const flattenedSourceObj = this.flattenObject(sourceObj);
    const flattenedTargetObj = this.flattenObject(targetObj);

    const sourceKeys = Object.keys(flattenedSourceObj);
    const targetKeys = Object.keys(flattenedTargetObj);

    const sourceEmbeddings = await this.calculateEmbeddings(sourceKeys);
    const targetEmbeddings = await this.calculateEmbeddings(targetKeys);

    const flattenedResult: ObjectWithStringKeys = {};

    for (let i = 0; i < sourceKeys.length; i++) {
      const sourceKey = sourceKeys[i];
      const sourceEmbedding = sourceEmbeddings[i];
      const closestTargetIndex = this.findClosestMatch(sourceEmbedding, targetEmbeddings);
      
      if (closestTargetIndex !== null) {
        const closestTargetKey = targetKeys[closestTargetIndex];
        flattenedResult[closestTargetKey] = flattenedSourceObj[sourceKey];
      }
    }

    return this.unflattenObject(flattenedResult) as U;
  }
}

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