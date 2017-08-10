package colman66.kakuro.server;

import static spark.Spark.*;
import spark.servlet.SparkApplication;

public class Main implements SparkApplication {
  public void init() {
    try {
      initSpark();
    } catch (Exception e) {
    }
  }

  public static void main(String... args) throws Exception {
    initSpark();
  }

  private static void initSpark() throws Exception {
    staticFiles.location("/public");
    post("/solveFromImage", new SolveFromImageHandler());
    post("/solveFromJson", new SolveFromJsonHandler());
  }
}