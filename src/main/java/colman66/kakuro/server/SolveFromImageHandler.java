package colman66.kakuro.server;

import com.fasterxml.jackson.core.*;
import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import javax.servlet.http.HttpServletRequest;
import org.apache.commons.fileupload.*;
import org.apache.commons.fileupload.disk.*;
import org.apache.commons.fileupload.servlet.*;
import org.apache.commons.io.*;
import spark.*;

public class SolveFromImageHandler implements Route {
  final ObjectMapper mapper = new ObjectMapper();

  ServletFileUpload uploadParser;

  public SolveFromImageHandler() throws IOException {
    DiskFileItemFactory factory = new DiskFileItemFactory();
    File uploadDir = Files.createTempDirectory(null).toFile();
    Runtime.getRuntime().addShutdownHook(new Thread() {
      public void run() {
        try {
          FileUtils.deleteDirectory(uploadDir);
        } catch (IOException e) {
        }
      }
    });
    factory.setRepository(uploadDir);
    uploadParser = new ServletFileUpload(factory);
  }

  public Object handle(Request request, Response response) throws Exception {
    boolean useInternalFormat = "internal".equals(request.queryParams("format"));
    HttpServletRequest raw = request.raw();
    List<FileItem> items = uploadParser.parseRequest(raw);

    FileItem image = items.stream().filter(p -> p.getFieldName().equals("image")).findAny().orElse(null);

    if (image == null || image.getSize() == 0) {
      response.status(400);
      return "Expected a POST form with enctype=\"multipart/form-data\" and a non-empty file with the key \"image\".";
    }

    JsonNode board = ImageProcessing.getBoardFromImage(image);
    KakuroSolver solver = new KakuroSolver((ArrayNode) board);
    response.type("application/json");

    if (useInternalFormat)
    {
          return mapper.writeValueAsString(solver.getResultJson());
    }

    return mapper.writeValueAsString(BoardConversions.appFromInternal(solver.getResultJson()));
  }
}