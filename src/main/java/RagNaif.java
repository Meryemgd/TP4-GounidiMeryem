import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.retriever.EmbeddingStoreRetriever;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;

public class RagNaif {

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) {
        try {
            // ====================================================================================
            // Phase 1 : Enregistrement des embeddings
            // ====================================================================================

            System.out.println("=== Phase 1 : Enregistrement des embeddings ===");

            // 1. Récupération du Path du fichier PDF
            Path pdfPath = Paths.get("src/main/resources/rag.pdf");
            System.out.println("Chargement du fichier : " + pdfPath);

            // 2. Création du parser PDF
            ApacheTikaDocumentParser parser = new ApacheTikaDocumentParser();

            // 3. Chargement du document
            Document document = FileSystemDocumentLoader.loadDocument(pdfPath, parser);
            System.out.println("Document chargé !");

            // 4. Création du DocumentSplitter et découpage en chunks
            DocumentSplitter splitter = DocumentSplitters.recursive(500, 0);
            List<TextSegment> segments = splitter.split(document);
            System.out.println("Document découpé en " + segments.size() + " segments");

            // 5. Création du modèle d'embedding
            EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

            // 6. Création des embeddings pour les segments
            List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
            System.out.println(embeddings.size() + " embeddings créés");

            // 7. Stockage dans un magasin d'embeddings en mémoire
            EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
            embeddingStore.addAll(embeddings, segments);
            System.out.println("Embeddings stockés dans le magasin");

            System.out.println("Phase 1 terminée avec succès!");

            // ====================================================================================
            // Phase 2 : Utilisation des embeddings pour répondre aux questions
            // ====================================================================================

            System.out.println("\n=== Phase 2 : Configuration de l'assistant RAG ===");

            // Création du ChatModel
            ChatLanguageModel chatModel = GoogleAiGeminiChatModel.builder()
                    .apiKey(System.getenv("GeminiKey"))
                    .modelName("gemini-2.5-flash")
                    .logRequestsAndResponses(true)
                    .temperature(0.2)
                    .build();

            // Création du ContentRetriever avec EmbeddingStoreRetriever
            EmbeddingStoreRetriever retriever = EmbeddingStoreRetriever.from(
                    embeddingStore,
                    embeddingModel,
                    3,
                    0.7     
            );

            // Ajout d'une mémoire pour 10 messages
            ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

            // Création de l'assistant avec le pattern builder
            Assistant assistant = AiServices.builder(Assistant.class)
                    .chatLanguageModel(chatModel)
                    .chatMemory(chatMemory)
                    .retriever(retriever)
                    .build();

            System.out.println("Assistant RAG configuré avec succès!");
            System.out.println("Vous pouvez maintenant poser vos questions...");

            // Boucle de questions-réponses
            Scanner scanner = new Scanner(System.in);
            while (true) {
                System.out.print("\nPosez votre question (ou 'exit' pour quitter) : ");
                String question = scanner.nextLine();

                if (question.equalsIgnoreCase("exit")) {
                    break;
                }

                if (question.trim().isEmpty()) {
                    continue;
                }

                try {
                    String response = assistant.chat(question);
                    System.out.println("\n Réponse : " + response);
                } catch (Exception e) {
                    System.out.println("\n Erreur lors de la génération de la réponse : " + e.getMessage());
                }
            }

            scanner.close();
            System.out.println("Au revoir !");

        } catch (Exception e) {
            System.err.println(" Erreur lors de l'exécution : " + e.getMessage());
            e.printStackTrace();
        }
    }
}