// code by jph
package ch.ethz.idsc.subare.ch05.racetrack;

import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.ArrayQ;
import ch.ethz.idsc.tensor.io.ResourceData;
import junit.framework.TestCase;

public class RacetrackTest extends TestCase {
  public void testStartAction() {
    Racetrack racetrack = new Racetrack(ResourceData.of("/ch05/track0.png"), 3);
    Index statesIndex = Index.build(racetrack.states());
    assertEquals(statesIndex.size(), 724);
    assertEquals(racetrack.statesStart, Tensors.fromString("{{0, 1, 0, 0}, {0, 2, 0, 0}, {0, 3, 0, 0}}"));
    assertEquals(racetrack.statesTerminal.length() % 3, 1);
    // System.out.println(racetrack.statesTerminal);
    // for (Tensor state : racetrack.statesStart)
    // System.out.println(racetrack.actions(state));
  }

  public void testSome() {
    Racetrack racetrack = new Racetrack(ResourceData.of("/ch05/track0.png"), 3);
    for (Tensor state : racetrack.states())
      racetrack.actions(state);
  }

  public void testArray() {
    Racetrack racetrack = new Racetrack(ResourceData.of("/ch05/track0.png"), 3);
    ArrayQ.require(racetrack.actions);
  }
}
