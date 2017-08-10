package colman66.kakuro.server;

import java.awt.geom.Ellipse2D;
import java.util.HashMap;
import java.util.Map;

import javax.net.ssl.ExtendedSSLSession;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.*;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import org.apache.commons.lang3.StringUtils;

public class BoardConversions {
    public static JsonNode internalFromApp(JsonNode input) {
        ObjectNode result = input.deepCopy();
        ArrayNode inputBoard = (ArrayNode) result.get("input");
        ArrayNode solvedBoard = (ArrayNode) result.get("solved");

        convertBoardToInternal(inputBoard);
        convertBoardToInternal(solvedBoard);

        return result;
    }

    public static JsonNode appFromInternal(JsonNode input) {
        ObjectNode result = input.deepCopy();
        ArrayNode inputBoard = (ArrayNode) result.get("input");
        ArrayNode solvedBoard = (ArrayNode) result.get("solved");

        convertBoardToApp(inputBoard);
        convertBoardToApp(solvedBoard);

        return result;

    }

    private static void convertBoardToApp(ArrayNode board) {
        for (int i = 0; i < board.size(); ++i) {
            ArrayNode row = (ArrayNode) board.get(i);
            for (int j = 0; j < row.size(); ++j) {
                JsonNode cell = row.get(j);

                String state = null, value = null, right = null, down = null;

                if (cell.isNumber()) {
                    state = "white";
                    value = cell.asText();
                } else if (cell.isNull()) {
                    state = "white";
                    value = "";
                } else if ("X".equals(cell.asText())) {
                    state = "black";
                } else if (cell.has("down")) {
                    down = cell.get("down").asText();
                    if (cell.has("right")) {
                        state = "downright";
                        right = cell.get("right").asText();
                    } else {
                        state = "down";
                    }
                } else if (cell.has("right")) {
                    state = "right";
                    right = cell.get("right").asText();
                }

                ObjectNode converted = JsonNodeFactory.instance.objectNode();
                if (state != null) {
                    converted.set("state", JsonNodeFactory.instance.textNode(state));
                }
                if (value != null) {
                    converted.set("value", JsonNodeFactory.instance.textNode(value));
                }
                if (right != null) {
                    converted.set("right", JsonNodeFactory.instance.textNode(right));
                }
                if (down != null) {
                    converted.set("down", JsonNodeFactory.instance.textNode(down));
                }

                row.set(j, converted);
            }
        }
    }

    private static void convertBoardToInternal(ArrayNode board) {
        for (int i = 0; i < board.size(); ++i) {
            ArrayNode row = (ArrayNode) board.get(i);
            for (int j = 0; j < row.size(); ++j) {
                ObjectNode cell = (ObjectNode)row.get(j);

                String state = cell.get("state").asText();
                JsonNode converted = null;

                switch(state)
                {
                    case "white" :
                        String value = cell.get("value").asText();
                        if (StringUtils.isNumeric(value))
                        {
                            converted = JsonNodeFactory.instance.numberNode(Integer.parseInt(value));
                        }
                        break;
                    case "black" :
                        converted = JsonNodeFactory.instance.textNode("X");
                        break;
                    case "right" :
                        converted = JsonNodeFactory.instance.objectNode()
                                .set("right", JsonNodeFactory.instance.numberNode(cell.get("right").asInt()));
                        break;
                    case "down" :
                        converted = JsonNodeFactory.instance.objectNode()
                                .set("down", JsonNodeFactory.instance.numberNode(cell.get("down").asInt()));                        break;
                    case "downright" :
                        converted = JsonNodeFactory.instance.objectNode()
                                .set("right", JsonNodeFactory.instance.numberNode(cell.get("right").asInt()));
                        ((ObjectNode)converted).set("down", JsonNodeFactory.instance.numberNode(cell.get("down").asInt()));
                        break;
                }

                row.set(j, converted);
            }
        }
    }
}