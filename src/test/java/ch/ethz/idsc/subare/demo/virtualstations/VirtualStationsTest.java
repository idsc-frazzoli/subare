// code by jph
package ch.ethz.idsc.subare.demo.virtualstations;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Total;
import ch.ethz.idsc.tensor.sca.Power;
import junit.framework.TestCase;

public class VirtualStationsTest extends TestCase {
  // each state the time interval followed by the NVnodes many virtual node informations
  // per time interval there are 2^NVnodes many different states (the end state is an additional interval)
  public void testStateSize() {
    VirtualStations virtualStations = (VirtualStations) VirtualStations.INSTANCE;
    virtualStations.states().forEach(v -> assertEquals(v.length(), virtualStations.getNVnodes() + 1));
    assertEquals(virtualStations.states().length(), (int) ((virtualStations.getTimeIntervals() + 1) * Math.pow(2, virtualStations.getNVnodes())));
  }

  // startStates all contain the lowest inveral number 0
  public void testStartState() {
    VirtualStations virtualStations = (VirtualStations) VirtualStations.INSTANCE;
    Tensor startStates = virtualStations.startStates();
    startStates.forEach(v -> assertEquals(v.Get(0), RealScalar.ZERO));
  }

  public void testActions() {
    VirtualStations virtualStations = (VirtualStations) VirtualStations.INSTANCE;
    for (Tensor state : virtualStations.states()) {
      Tensor actions = virtualStations.actions(state);
      if (virtualStations.isTerminal(state)) {
        assertEquals(actions.length(), 1);
      } else {
        Scalar expected = Power.of(RealScalar.of(2),
            Total.ofVector(state.extract(1, state.length())).multiply(RealScalar.of(virtualStations.getNVnodes() - 1)));
        Scalar actual = RealScalar.of(actions.length());
        assertEquals(expected, actual);
      }
    }
  }

  public void testTerminalStates() {
    VirtualStations virtualStations = (VirtualStations) VirtualStations.INSTANCE;
    for (Tensor state : virtualStations.states()) {
      assertEquals(virtualStations.isTerminal(state), state.Get(0).equals(RealScalar.of(virtualStations.getTimeIntervals())));
    }
  }
}
