import express from "express";
import path from "path";
import fs from "fs";
import { exec } from "child_process";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

// Initialize Gemini SDK
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      "User-Agent": "aistudio-build",
    },
  },
});

app.use(express.json());

// API routes FIRST
// 1. Help & Healthcheck
app.get("/api/health", (req, res) => {
  res.json({ status: "ok" });
});

// 2. CAD Query with Gemini
async function generateCADContent(prompt: string, systemInstruction: string) {
  const models = ["gemini-3.5-flash", "gemini-3.1-flash-lite"];
  let lastError: any = null;

  for (const model of models) {
    // Retry up to 2 times for each model
    for (let attempt = 1; attempt <= 2; attempt++) {
      try {
        console.log(`Querying Gemini with model ${model} (attempt ${attempt})...`);
        const response = await ai.models.generateContent({
          model,
          contents: prompt,
          config: {
            systemInstruction,
            responseMimeType: "application/json",
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                title: {
                  type: Type.STRING,
                  description: "Short descriptive title of the CAD model",
                },
                explanation: {
                  type: Type.STRING,
                  description: "Brief explanation of how the geometry is constructed",
                },
                python_code: {
                  type: Type.STRING,
                  description: "The complete, valid Python script using cad_builder library.",
                },
                parameters: {
                  type: Type.ARRAY,
                  description: "List of customizable parameters extracted from the code.",
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      name: {
                        type: Type.STRING,
                        description: "Variable name used in get_param",
                      },
                      label: {
                        type: Type.STRING,
                        description: "User-friendly display label (e.g. 'Shaft Length')",
                      },
                      type: {
                        type: Type.STRING,
                        description: "'float' or 'int'",
                      },
                      default: {
                        type: Type.NUMBER,
                        description: "Default value",
                      },
                      min: {
                        type: Type.NUMBER,
                        description: "Minimum sensible value",
                      },
                      max: {
                        type: Type.NUMBER,
                        description: "Maximum sensible value",
                      },
                      description: {
                        type: Type.STRING,
                        description: "Short tooltip or description for the user",
                      },
                    },
                    required: ["name", "label", "type", "default", "min", "max", "description"],
                  },
                },
              },
              required: ["title", "explanation", "python_code", "parameters"],
            },
          },
        });

        if (response && response.text) {
          return JSON.parse(response.text.trim());
        }
        throw new Error("Empty response received from Gemini");
      } catch (err: any) {
        lastError = err;
        console.warn(`Attempt ${attempt} with model ${model} failed:`, err.message || err);
        // Short delay before retrying
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    }
  }

  throw lastError || new Error("Failed to generate CAD design after retries and fallbacks");
}

app.post("/api/cad/query", async (req, res) => {
  try {
    const { prompt } = req.body;
    if (!prompt) {
      res.status(400).json({ error: "Prompt is required" });
      return;
    }

    const systemInstruction = `You are an expert 3D CAD designer and code generator for our custom Python CAD library \`cad_builder\`.
Your task is to translate natural language queries into clean, parameterized Python CAD code that uses our library to construct beautiful 3D shapes.

Our library \`cad_builder\` provides these main classes and functions:
- \`cad_builder.get_param(name, default_value)\`: Reads a parameter value. Always use this to read variables that the user might want to adjust.
- \`cad_builder.Box(width, height, depth)\`: Creates a box centered at the origin.
- \`cad_builder.Cylinder(radius, height, sections=32)\`: Creates a cylinder along the Z-axis, centered at the origin.
- \`cad_builder.Sphere(radius, sections=32)\`: Creates a sphere centered at the origin.
- \`cad_builder.Cone(radius1, radius2, height, sections=32)\`: Creates a cone along the Z-axis (radius1 is bottom, radius2 is top).
- \`cad_builder.Torus(major_radius, minor_radius, sections=32)\`: Creates a torus centered at the origin.
- \`cad_builder.Hexagon(width, height)\`: Creates a 3D hexagon prism (width across flats, height).
- \`cad_builder.ThreadedShaft(diameter, length, pitch=1.5)\`: Creates a threaded-like cylinder shaft (diameter, length).
- \`cad_builder.Union(shape_list)\` or \`shape1 + shape2\`: Merges shapes together.
- \`cad_builder.Difference(shape1, shape2)\` or \`shape1 - shape2\`: Subtracts shape2 from shape1.
- \`cad_builder.Intersection(shape1, shape2)\` or \`shape1 & shape2\`: Intersects shapes.
- Shapes have transformation methods that return the modified shape:
  - \`.translate(x, y, z)\`
  - \`.rotate_x(angle_degrees)\`
  - \`.rotate_y(angle_degrees)\`
  - \`.rotate_z(angle_degrees)\`
- You can create helper prebuilt assemblies by combining primitives:
  - \`cad_builder.create_hex_bolt(diameter, length, head_width, head_height)\`: Generates a hex bolt assembly.
  - \`cad_builder.create_nut(diameter, thickness, outer_width, hole_tolerance=0.4)\`: Generates a hex nut with a threaded hole.
  - \`cad_builder.create_bracket(width, height, depth, hole_diameter, thickness=3.0)\`: Generates an L-bracket.
  - \`cad_builder.create_gear(teeth, thickness, outer_radius, inner_radius, bore_diameter)\`: Generates a spur gear.
- To export the design, you MUST call \`cad_builder.add_shape(shape)\` with the final shape or assembly.

Guidelines:
1. Always parameterize your code. Use \`cad_builder.get_param('param_name', default_value)\` for dimensions.
2. Provide a clean explanation of the geometric construction.
3. Keep the code clean, robust, and correctly aligned. Align shapes using translation so they join beautifully. For example, to place a bolt head on top of a shaft of length L, translate the head by Z = L/2 + head_height/2 if centered, or handle their origins carefully. All primitives are centered at the origin by default.
4. Ensure the returned parameter bounds are realistic (e.g. min > 0, max is sensible, default is in between).`;

    const data = await generateCADContent(prompt, systemInstruction);
    res.json(data);
  } catch (error: any) {
    console.error("Gemini query error:", error);
    res.status(500).json({ error: error.message || "Failed to generate CAD design" });
  }
});

// 3. Execute CAD python code and return 3D mesh
app.post("/api/cad/run", async (req, res) => {
  const { code, params } = req.body;
  if (!code) {
    res.status(400).json({ error: "Python code is required" });
    return;
  }

  const runId = Math.random().toString(36).substring(2, 10);
  const tempScriptPath = path.join(process.cwd(), `temp_cad_${runId}.py`);
  const tempParamsPath = path.join(process.cwd(), `temp_params_${runId}.json`);

  try {
    // Write code and parameters to files
    fs.writeFileSync(tempScriptPath, code);
    fs.writeFileSync(tempParamsPath, JSON.stringify(params || {}));

    // Run python script with correct environment variables and 50MB buffer
    const cmd = `python3 temp_cad_${runId}.py`;
    const env = {
      ...process.env,
      CAD_PARAMS_FILE: tempParamsPath,
    };

    exec(cmd, { env, maxBuffer: 1024 * 1024 * 50 }, (error, stdout, stderr) => {
      // Clean up files immediately
      try {
        if (fs.existsSync(tempScriptPath)) fs.unlinkSync(tempScriptPath);
        if (fs.existsSync(tempParamsPath)) fs.unlinkSync(tempParamsPath);
      } catch (err) {
        console.error("Cleanup error:", err);
      }

      if (error) {
        console.error("Python execution error:", stderr);
        res.status(400).json({
          error: "Python execution failed",
          details: stderr || error.message,
        });
        return;
      }

      try {
        // Parse python output. The script is designed to output a JSON at the end
        // Let's find the last line or find JSON structure in stdout
        const stdoutStr = stdout.toString();
        const jsonStartIndex = stdoutStr.indexOf("===CAD_OUTPUT_START===");
        const jsonEndIndex = stdoutStr.indexOf("===CAD_OUTPUT_END===");

        if (jsonStartIndex === -1 || jsonEndIndex === -1) {
          // If no delimiters, try parsing entire stdout as JSON
          try {
            const data = JSON.parse(stdoutStr);
            res.json(data);
          } catch (e) {
            res.status(400).json({
              error: "Python script did not output valid CAD JSON data",
              details: stdoutStr,
            });
          }
          return;
        }

        const jsonStr = stdoutStr.substring(
          jsonStartIndex + "===CAD_OUTPUT_START===".length,
          jsonEndIndex
        );
        const data = JSON.parse(jsonStr.trim());
        res.json(data);
      } catch (parseError: any) {
        console.error("JSON parse error from Python output:", parseError);
        res.status(500).json({
          error: "Failed to parse CAD data from script output",
          details: stdout.toString(),
        });
      }
    });
  } catch (err: any) {
    console.error("API execution crash:", err);
    // Cleanup on crash
    try {
      if (fs.existsSync(tempScriptPath)) fs.unlinkSync(tempScriptPath);
      if (fs.existsSync(tempParamsPath)) fs.unlinkSync(tempParamsPath);
    } catch (_) {}
    res.status(500).json({ error: err.message || "Failed to execute CAD script" });
  }
});

// Vite middleware and Static Serving setup
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
