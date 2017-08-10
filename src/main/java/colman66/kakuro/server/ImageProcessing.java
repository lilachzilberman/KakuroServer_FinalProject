package colman66.kakuro.server;

import com.fasterxml.jackson.core.*;
import com.fasterxml.jackson.databind.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import org.apache.commons.fileupload.*;
import org.apache.commons.io.*;
import org.apache.commons.io.input.*;

public class ImageProcessing {
  final static ObjectMapper mapper = new ObjectMapper();
  static boolean isBoolEnvSet(String name) {
    Map<String, String> env = System.getenv();
    String raw = env.get(name);
    return "true".equalsIgnoreCase(raw) || "1".equals(raw);
  }

  public static JsonNode getBoardFromImage(FileItem image) throws JsonProcessingException, IOException, Exception
  {
    Map<String, String> env = System.getenv();
    boolean imageViaFile = isBoolEnvSet("KAKURO_PYTHON_IMAGE_VIA_FILE");
    boolean keepImage = isBoolEnvSet("KAKURO_PYTHON_KEEP_IMAGE_FILES");
    String pythonScript = env.get("KAKURO_PYTHON_MAIN");
    String pythonBin = env.containsKey("PYTHON3_BIN") ? env.get("PYTHON3_BIN") : "python3";
    if (pythonScript == null) {
      return mapper.readTree("[[\"X\"]]");
    }

    if (!Files.exists(FileSystems.getDefault().getPath(pythonScript))) {
      throw new RuntimeException("Expected KAKURO_PYTHON_MAIN (" + pythonScript + ") to be a file");
    }

    Vector<String> pythonCmd = new Vector<String>();
    File imageAsFile = null;
    pythonCmd.add(pythonBin);
    pythonCmd.add(pythonScript);
    if (imageViaFile) {
      imageAsFile = writeUploadToDisk(image);
      pythonCmd.add(imageAsFile.getAbsolutePath());
    }
    try {
      ProcessBuilder pb = new ProcessBuilder(pythonCmd);
      pb.redirectError(ProcessBuilder.Redirect.INHERIT);
      System.out.println("Starting python script with:\n\t" + String.join(" ", pythonCmd));
      Process p = pb.start();
      if (!imageViaFile) {
        OutputStream pythonStdin = p.getOutputStream();
        IOUtils.copy(image.getInputStream(), pythonStdin);
        pythonStdin.flush();
        pythonStdin.close();
      }
      TeeInputStream split = new TeeInputStream(p.getInputStream(), System.out);
      return mapper.readTree(split);
    } finally {
      if (imageAsFile != null) {
        imageAsFile.deleteOnExit(); // long term cleanup if the delete() call fails or keepImage == true
        if (!keepImage) {
          imageAsFile.delete();
        }
      }
    }
  }

  static File writeUploadToDisk(FileItem image) throws Exception {
    String originalName = image.getName();
    if ("".equals(originalName) || originalName == null) {
      originalName = "C:\\kakuro.jpg";
    }
    String prefix = FilenameUtils.getBaseName(originalName);
    String suffix = "." + FilenameUtils.getExtension(originalName);
    if (".".equals(suffix)) {
      suffix = "jpg";
    }
    File tempFile = Files.createTempFile(prefix, suffix).toFile();
    image.write(tempFile);
    return tempFile;
  }
}