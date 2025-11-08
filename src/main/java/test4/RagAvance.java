package test4;


import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;
import dev.langchain4j.rag.query.transformer.ExpandingQueryTransformer;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

public class RagAvance {

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) {
        configureLogger();
        
        System.out.println("=== Test 4 : RAG Avancé ===");

        // Création du parser et du modèle d'embedding
        DocumentParser parser = new ApacheTikaDocumentParser();
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // Chargement et découpage du document avec filtrage
        List<TextSegment> segments = loadAndSplit("src/main/resources/rag.pdf", parser);
        
        segments = segments.stream()
                .filter(segment -> segment.text() != null && 
                                 segment.text().trim().length() > 10 &&
                                 !segment.text().trim().matches("\\s*"))
                .collect(Collectors.toList());

        System.out.println("Segments filtrés : " + segments.size());

        if (segments.isEmpty()) {
            System.err.println("Erreur : Aucun segment valide trouvé !");
            return;
        }

        // Création du magasin d'embeddings
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        try {
            System.out.println("Génération des embeddings...");
            embeddingStore.addAll(embeddingModel.embedAll(segments).content(), segments);
        } catch (Exception e) {
            System.err.println("Erreur lors de la génération des embeddings : " + e.getMessage());
            return;
        }

        // Modèle de chat Gemini
        String GEMINI_API_KEY = System.getenv("GeminiKey");
        if (GEMINI_API_KEY == null) {
            throw new IllegalStateException("Variable d'environnement GeminiKey manquante !");
        }

        ChatLanguageModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(GEMINI_API_KEY)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // === TECHNIQUE 1: Query Transformation ===
        CompressingQueryTransformer compressingTransformer = new CompressingQueryTransformer(chatModel);
        ExpandingQueryTransformer expandingTransformer = new ExpandingQueryTransformer(chatModel);

        // === TECHNIQUE 2: Content Retriever avec Re-ranking ===
        ContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(5) // Plus de résultats pour le re-ranking
                .minScore(0.3) // Score plus permissif
                .build();

        // === TECHNIQUE 3: Augmentateur avancé avec transformations ===
        DefaultRetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryTransformer(compressingTransformer) // Compression des requêtes
                .contentRetriever(retriever)
                .build();

        // Assistant avec RAG avancé
        Assistant assistantAvance = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(augmentor)
                .build();

        // === TECHNIQUE 4: Assistant basique pour comparaison ===
        ContentRetriever retrieverBasique = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        DefaultRetrievalAugmentor augmentorBasique = DefaultRetrievalAugmentor.builder()
                .contentRetriever(retrieverBasique)
                .build();

        Assistant assistantBasique = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(augmentorBasique)
                .build();


        // Interface utilisateur avec choix du mode
        Scanner scanner = new Scanner(System.in);
        System.out.println("\nChoisissez le mode :");
        System.out.println("1. Assistant RAG Avancé (avec techniques avancées)");
        System.out.println("2. Assistant RAG Basique (pour comparaison)");
        System.out.println("3. Comparaison automatique");
        
        while (true) {
            System.out.print("\nMode (1/2/3) ou 'exit' : ");
            String mode = scanner.nextLine().trim();
            
            if (mode.equalsIgnoreCase("exit")) break;
            
            if (!mode.matches("[123]")) {
                System.out.println("Mode invalide. Choisissez 1, 2, 3 ou 'exit'");
                continue;
            }
            
            System.out.print("Votre question : ");
            String question = scanner.nextLine();
            
            if (question.trim().isEmpty()) continue;

            try {
                switch (mode) {
                    case "1":
                        System.out.println("\n RAG AVANCÉ :");
                        String reponseAvancee = assistantAvance.chat(question);
                        System.out.println("Gemini (Avancé) : " + reponseAvancee);
                        break;
                        
                    case "2":
                        System.out.println("\n RAG BASIQUE :");
                        String reponseBasique = assistantBasique.chat(question);
                        System.out.println("Gemini (Basique) : " + reponseBasique);
                        break;
                        
                    case "3":
                        System.out.println("\n RAG AVANCÉ :");
                        String avancee = assistantAvance.chat(question);
                        System.out.println("Gemini (Avancé) : " + avancee);
                        
                        System.out.println("\n RAG BASIQUE :");
                        String basique = assistantBasique.chat(question);
                        System.out.println("Gemini (Basique) : " + basique);
                        
                        System.out.println("\n COMPARAISON TERMINÉE");
                        break;
                }
            } catch (Exception e) {
                System.err.println("Erreur lors de la réponse : " + e.getMessage());
            }
        }
        
        scanner.close();
    }

    private static List<TextSegment> loadAndSplit(String chemin, DocumentParser parser) {
        Path path = Paths.get(chemin);
        Document doc = FileSystemDocumentLoader.loadDocument(path, parser);
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        return splitter.split(doc);
    }
}
