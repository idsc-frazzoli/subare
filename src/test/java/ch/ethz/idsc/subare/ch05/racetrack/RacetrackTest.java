// code by jph
package ch.ethz.idsc.subare.ch05.racetrack;

import java.util.Arrays;

import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.ArrayQ;
import ch.ethz.idsc.tensor.alg.Dimensions;
import ch.ethz.idsc.tensor.io.ResourceData;
import junit.framework.TestCase;

public class RacetrackTest extends TestCase {
  public void testStartAction() {
    Racetrack racetrack = new Racetrack(ResourceData.of("/ch05/track0.png"), 3);
    Index statesIndex = Index.build(racetrack.states());
    assertEquals(statesIndex.size(), 724);
    assertEquals(racetrack.statesStart, Tensors.fromString("{{1, 0, 0, 0}, {2, 0, 0, 0}, {3, 0, 0, 0}}"));
    assertEquals(racetrack.statesTerminal.length() % 3, 1);
    try {
      racetrack.actions(Tensors.vector(1, 0, 0));
      fail();
    } catch (Exception exception) {
      // ---
    }
  }

  public void testMove() {
    Racetrack racetrack = new Racetrack(ResourceData.of("/ch05/track0.png"), 3);
    assertEquals(Dimensions.of(racetrack.image()), Arrays.asList(8, 11, 4));
    Tensor start = Tensors.vector(1, 0, 0, 0);
    assertTrue(racetrack.isStart(start));
    assertFalse(racetrack.isTerminal(start));
    Tensor next = racetrack.integrate(start, Tensors.vector(1, 1)); // vy
    assertTrue(racetrack.statesIndex.containsKey(next));
    Tensor move = racetrack.move(start, Tensors.vector(2, 3));
    assertEquals(move, Tensors.vector(3, 3, 2, 3));
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
