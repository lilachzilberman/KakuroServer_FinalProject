import org.testng.annotations.*;
import colman66.kakuro.server.KakuroSolver;
import com.fasterxml.jackson.core.*;
import com.fasterxml.jackson.databind.*;
import java.io.IOException;
import java.io.File;
import java.net.URL;
import java.net.URISyntaxException;
import static org.testng.Assert.*;
import java.nio.file.Path;
import com.fasterxml.jackson.databind.node.ArrayNode;
import org.testng.ITest;

public final class BoardFileTest implements ITest
{
    public String getTestName() {
        return basename;
    }

    private String basename;
    final ObjectMapper mapper = new ObjectMapper();

    private static File urlToFile(URL url) {
        try {
            return new File(url.toURI());
        } catch(URISyntaxException e) {
            return null;
        }
    }

    public BoardFileTest(String basename) {
        this.basename = basename;
    }

    @Test
    public void boardTest() throws JsonProcessingException, IOException {
        URL inputFileUrl = this.getClass().getResource(this.basename + ".json");
        JsonNode input, output;
        KakuroSolver solver;
        try {
            input = mapper.readTree(inputFileUrl);
            solver = new KakuroSolver((ArrayNode) input);
            output = solver.getResultJson();
        } catch (Throwable e) {
            if (this.basename.endsWith("." + e.getClass().getSimpleName())) {
                return;
            }
            throw e;
        }
        URL snapshotFileUrl = this.getClass().getResource(this.basename + ".output.json");
        if (snapshotFileUrl == null) {
            Path inputPath = urlToFile(inputFileUrl).toPath();
            Path buildPath = inputPath
                .getParent()
                .getParent()
                .getParent();
            Path testPath = buildPath
                .resolveSibling("src")
                .resolve("test");
            File outputFile = testPath
                .resolve("resources")
                .resolve(this.basename + ".output.json")
                .toFile();
            outputFile.createNewFile();
            mapper.enable(SerializationFeature.INDENT_OUTPUT)
                .writeValue(outputFile, output);
        } else {
            JsonNode expected = mapper.readTree(snapshotFileUrl);
            assert(expected.equals(output));
        }
    }
}