package colman66.kakuro.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.*;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;

import org.chocosolver.solver.*;
import org.chocosolver.solver.search.strategy.Search;
import org.chocosolver.solver.exception.ContradictionException;
import org.chocosolver.solver.variables.IntVar;

import java.util.Vector;
import java.util.Collection;

public final class KakuroSolver {
    private ArrayNode board;
    private JsonNode solvedBoard;
    private Model model;
    private IntVar[][] vars;
    private Vector<IntVar> currentSumVars = new Vector<IntVar>();
    private boolean considerNumberCells = true;
    private String status;
    
    public KakuroSolver(ArrayNode board) {
        this.board = board;
        solve();
        if (!this.solvedBoard.isArray()) {
            considerNumberCells = false;
            solve();
        }
    }
    
    public JsonNode getResultJson() {
        ObjectNode result = JsonNodeFactory.instance.objectNode();

        result.put("input", board);
        result.put("solved", solvedBoard);
        result.put("status", status);
        
        return result;
    }

    public JsonNode getSolvedBoardJson() {
        return this.solvedBoard;
    }

    public JsonNode getBoardJson() {
        return this.board;
    }

    private void solve() {
        this.status = "NOT_SOLVED";
        this.solvedBoard = JsonNodeFactory.instance.nullNode();

        this.model = new Model("Kakuro board");
        this.vars = model.intVarMatrix("k", numOfRows(), numOfCols(), 1, 9);
        try {
            for (int i = 0; i < board.size(); ++i) {
                ArrayNode row = (ArrayNode) board.get(i);
                for (int j = 0; j < row.size(); ++j) {
                    JsonNode cell = row.get(j);
                    int sumRight = beginSum(cell, "right");
                    if (sumRight > 0) {
                        for (int jj = j + 1; jj < row.size(); ++jj) {
                            if (!addCellToModel(i, jj)) break;
                        }
                        finishSum(sumRight);
                    }
                    int sumDown = beginSum(cell, "down");
                    if (sumDown > 0) {
                        for (int ii = i + 1; ii < numOfRows(); ++ii) {
                            if (!addCellToModel(ii, j)) break;
                        }
                        finishSum(sumDown);
                    }
                }
            }
        }
        catch (ContradictionException e) {
            return;
        }
        Solver solver = model.getSolver();
        solver.setSearch(Search.defaultSearch(model));
        if (!solver.solve()) {
            return;
        }

        this.solvedBoard = board.deepCopy();
        this.status = this.considerNumberCells ? "SOLVED" : "SOLVED_DIFFERENTLY";

        for (int i = 0; i < solvedBoard.size(); ++i) {
            ArrayNode row = (ArrayNode) solvedBoard.get(i);
            for (int j = 0; j < row.size(); ++j) {
                JsonNode cell = row.get(j);
                if (!cellIsVariable(cell)) {
                    continue;
                }
                row.set(j, JsonNodeFactory.instance.numberNode(vars[i][j].getValue()));
            }
        }
    }

    private int numOfCols() {
        if (board.size() == 0) {
            return 0;
        }
        int cols = 0;
        for (int i = 0; i < board.size(); ++i) {
            ArrayNode row = (ArrayNode) board.get(i);
            if (i == 0) {
                cols = row.size();
            } else {
                assert( cols == row.size() );
            }
        }
        return cols;
    }

    private int numOfRows() {
        return board.size();
    }

    private boolean cellIsVariable(JsonNode cell) {
        return !(cell.isObject() || cell.isTextual());
    }

    private boolean addCellToModel(int i, int j) throws ContradictionException {
        JsonNode cell = cell(i, j);
        if (cellIsVariable(cell)) {
            currentSumVars.add(vars[i][j]);
            if (cell.isNumber() && considerNumberCells) {
                vars[i][j].instantiateTo(cell.asInt(), Cause.Null);
            }
            return true;
        }
        return false;
    }
    
    private JsonNode cell(int i, int j) {
        return ((ArrayNode) board.get(i)).get(j);
    }

    private int beginSum(JsonNode cell, String key) {
        currentSumVars.clear();
        if (!cell.isObject()) {
            return 0;
        }
        return cell.has(key) ? cell.get(key).asInt(0) : 0;
    }

    private void finishSum(int sum) {
        IntVar[] elemArray = new IntVar[0];
        elemArray = currentSumVars.toArray(elemArray);
        if (elemArray.length > 0) {
            model.sum(elemArray.clone(), "=", sum).post();
            model.allDifferent(elemArray.clone()).post();
        }
        currentSumVars.clear();
    }
}