// code by jph
package ch.ethz.idsc.subare.ch05.racetrack;

import java.io.File;
import java.io.IOException;
import java.util.zip.DataFormatException;

import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.io.Import;
import junit.framework.TestCase;

public class RacetrackTest extends TestCase {
  public void testStartAction() throws ClassNotFoundException, DataFormatException, IOException {
    String path = "".getClass().getResource("/ch05/track0.png").getPath();
    Racetrack racetrack = new Racetrack(Import.of(new File(path)), 3);
    Index statesIndex = Index.build(racetrack.states());
    // assertEquals(racetrack.statesStart, Tensors.fromString("{{0, 1, 0, 0}, {0, 2, 0, 0}, {0, 3, 0, 0}}"));
    assertEquals(racetrack.statesTerminal.length() % 3, 1);
    // System.out.println(racetrack.statesTerminal);
    // for (Tensor state : racetrack.statesStart)
    // System.out.println(racetrack.actions(state));
  }

  public void testSome() throws ClassNotFoundException, DataFormatException, IOException {
    String path = "".getClass().getResource("/ch05/track0.png").getPath();
    Racetrack racetrack = new Racetrack(Import.of(new File(path)), 3);
    for (Tensor state : racetrack.states())
      racetrack.actions(state);
  }
}
