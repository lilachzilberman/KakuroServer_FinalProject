package colman66.kakuro.server;

import spark.*;
import com.fasterxml.jackson.databind.node.*;
import com.fasterxml.jackson.databind.*;


public class SolveFromJsonHandler implements Route {
  final ObjectMapper mapper = new ObjectMapper();

  public Object handle(Request request, Response response) throws Exception {
    if (!request.contentType().equals("application/json")) {
      response.status(400);
      return "Expected JSON input";
    }
    JsonNode boardJson = mapper.readTree(request.body());
    KakuroSolver solver = new KakuroSolver((ArrayNode) boardJson);
    response.type("application/json");
    return mapper.writeValueAsString(solver.getResultJson());
  }
}