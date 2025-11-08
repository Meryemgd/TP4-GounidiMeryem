package test3;


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
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

public class TestRoutage {

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) {
        configureLogger();
        
        System.out.println("=== Test 3 : Routage ===");

        // Création du parser et du modèle d'embedding
        DocumentParser parser = new ApacheTikaDocumentParser();
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // Chargement et découpage des documents avec filtrage des segments vides
        List<TextSegment> segmentsIA = loadAndSplit("src/main/resources/rag.pdf", parser);
        List<TextSegment> segmentsRecettes = loadAndSplit("src/main/resources/Easy_recipes.pdf", parser);

        // FILTRE PLUS RIGOUREUX DES SEGMENTS VIDES
        segmentsIA = segmentsIA.stream()
                .filter(segment -> segment.text() != null && 
                                 segment.text().trim().length() > 10 && // Au moins 10 caractères
                                 !segment.text().trim().matches("\\s*")) // Pas seulement des espaces
                .collect(Collectors.toList());
        
        segmentsRecettes = segmentsRecettes.stream()
                .filter(segment -> segment.text() != null && 
                                 segment.text().trim().length() > 10 && // Au moins 10 caractères
                                 !segment.text().trim().matches("\\s*")) // Pas seulement des espaces
                .collect(Collectors.toList());

        System.out.println("Segments IA (filtrés) : " + segmentsIA.size());
        System.out.println("Segments Recettes (filtrés) : " + segmentsRecettes.size());

        // Vérification qu'il y a des segments valides
        if (segmentsIA.isEmpty()) {
            System.err.println("Erreur : Aucun segment IA valide trouvé !");
            return;
        }
        if (segmentsRecettes.isEmpty()) {
            System.err.println("Erreur : Aucun segment recette valide trouvé !");
            return;
        }

        // Création des magasins d'embeddings séparés
        EmbeddingStore<TextSegment> storeIA = new InMemoryEmbeddingStore<>();
        EmbeddingStore<TextSegment> storeRecettes = new InMemoryEmbeddingStore<>();

        try {
            // Génération des embeddings avec gestion d'erreur
            System.out.println("Génération des embeddings IA...");
            storeIA.addAll(embeddingModel.embedAll(segmentsIA).content(), segmentsIA);
            
            System.out.println("Génération des embeddings Recettes...");
            storeRecettes.addAll(embeddingModel.embedAll(segmentsRecettes).content(), segmentsRecettes);
            
        } catch (Exception e) {
            System.err.println("Erreur lors de la génération des embeddings : " + e.getMessage());
            return;
        }

        ContentRetriever retrieverIA = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeIA)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.5)
                .build();

        ContentRetriever retrieverRecettes = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeRecettes)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.5)
                .build();

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

        // Configuration du routage avec descriptions
        Map<ContentRetriever, String> retrieverDescriptions = new HashMap<>();
        retrieverDescriptions.put(retrieverIA, "Documents de cours sur le RAG, le fine-tuning et l'intelligence artificielle");
        retrieverDescriptions.put(retrieverRecettes, "Document sur les recettes de cuisine faciles, les ingrédients et les instructions de préparation");

        // Création du routeur de requêtes
        LanguageModelQueryRouter queryRouter = new LanguageModelQueryRouter(chatModel, retrieverDescriptions);

        // Création de l'augmentateur avec routage
        DefaultRetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // Création de l'assistant avec routage
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(augmentor)
                .build();

        System.out.println(" Assistant RAG avec routage prêt !");

        // Interaction avec l'utilisateur
        Scanner scanner = new Scanner(System.in);
        System.out.println("\nPosez votre question (ou 'exit' pour quitter) :");
        System.out.println("Exemples : 'Qu'est-ce que le RAG ?' ou 'Comment faire une recette simple ?'");
        
        while (true) {
            System.out.print("\nVous : ");
            String question = scanner.nextLine();
            if (question.equalsIgnoreCase("exit")) break;

            try {
                String reponse = assistant.chat(question);
                System.out.println("Gemini : " + reponse);
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
