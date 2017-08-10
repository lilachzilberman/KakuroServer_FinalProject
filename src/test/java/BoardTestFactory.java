import org.testng.annotations.*;

public class BoardTestFactory 
{
    @Factory
    public Object[] testAllBoards() {
        return new Object[] {
            new BoardFileTest("2Solutions"),
            new BoardFileTest("empty1"),
            new BoardFileTest("Kak1"),
            new BoardFileTest("Kak2"),
            new BoardFileTest("Kak3"),
            new BoardFileTest("Ksenia"),
            new BoardFileTest("noSolution1"),
            new BoardFileTest("noSolution2"),
            new BoardFileTest("noSolution3"),
            new BoardFileTest("noSolution4"),
            new BoardFileTest("badSolution1"),
            new BoardFileTest("badSolution2"),
            new BoardFileTest("badSolution3"),
            new BoardFileTest("badSolution4"),
            new BoardFileTest("badBoard1.AssertionError"),
            new BoardFileTest("badBoard2.AssertionError")
        };
    }
}