package test5;


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
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

public class RagWebSearch {

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) {
        configureLogger();

        System.out.println("=== Test 5 : RAG avec Recherche Web ===");

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

        // Création du magasin d'embeddings pour les documents locaux
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        try {
            System.out.println("Génération des embeddings...");
            embeddingStore.addAll(embeddingModel.embedAll(segments).content(), segments);
        } catch (Exception e) {
            System.err.println("Erreur lors de la génération des embeddings : " + e.getMessage());
            return;
        }

        // Configuration des modèles
        String GEMINI_API_KEY = System.getenv("GeminiKey");
        String TAVILY_API_KEY = System.getenv("TavilyKey");

        if (GEMINI_API_KEY == null) {
            throw new IllegalStateException("Variable d'environnement GeminiKey manquante !");
        }

        if (TAVILY_API_KEY == null) {
            throw new IllegalStateException("Variable d'environnement TavilyKey manquante ! Obtenez une clé sur https://tavily.com/");
        }

        ChatLanguageModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(GEMINI_API_KEY)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // === CRÉATION DU RAG HYBRIDE (DOCUMENTS + WEB) ===

        // 1. ContentRetriever pour les documents locaux
        ContentRetriever documentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.5)
                .build();

        // 2. WebSearchEngine avec Tavily
        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(TAVILY_API_KEY)
                .build();

        // 3. ContentRetriever pour la recherche web
        ContentRetriever webRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .maxResults(3)
                .build();

        // 4. QueryRouter pour utiliser les 2 ContentRetrievers
        DefaultQueryRouter queryRouter = new DefaultQueryRouter(documentRetriever, webRetriever);

        // 5. RetrievalAugmentor avec le QueryRouter
        DefaultRetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // 6. Assistant avec RAG hybride
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(augmentor)
                .build();

        System.out.println(" Assistant RAG hybride (Documents + Web) prêt !");
        System.out.println(" Sources utilisées :");
        System.out.println("  • Documents PDF locaux (rag.pdf)");
        System.out.println("  • Recherche web via Tavily");
        System.out.println("  • Routeur automatique entre les sources");

        // Interface utilisateur
        Scanner scanner = new Scanner(System.in);
        System.out.println("\nPosez votre question (ou 'exit' pour quitter) :");
        System.out.println(" Suggestions :");
        System.out.println("  - Questions sur le RAG (documents locaux)");
        System.out.println("  - Questions d'actualité (recherche web)");
        System.out.println("  - Questions mixtes (combinaison des sources)");

        while (true) {
            System.out.print("\nVous : ");
            String question = scanner.nextLine();

            if (question.equalsIgnoreCase("exit")) break;
            if (question.trim().isEmpty()) continue;

            try {
                System.out.println("\n Recherche en cours (documents + web)...");
                long startTime = System.currentTimeMillis();

                String reponse = assistant.chat(question);

                long responseTime = System.currentTimeMillis() - startTime;

                System.out.println(" Assistant RAG : " + reponse);
                System.out.printf(" Temps de réponse : %d ms\n", responseTime);

            } catch (Exception e) {
                System.err.println(" Erreur lors de la réponse : " + e.getMessage());
            }
        }

        scanner.close();
        System.out.println("Au revoir !");
    }

    private static List<TextSegment> loadAndSplit(String chemin, DocumentParser parser) {
        Path path = Paths.get(chemin);
        Document doc = FileSystemDocumentLoader.loadDocument(path, parser);
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        return splitter.split(doc);
    }
}