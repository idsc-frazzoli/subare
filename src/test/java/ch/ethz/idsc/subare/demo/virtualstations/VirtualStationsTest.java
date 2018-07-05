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
    VirtualStations vs = new VirtualStations();
    vs.states().forEach(v -> assertEquals(v.length(), vs.getNVnodes() + 1));
    assertEquals(vs.states().length(), (int) ((vs.getTimeIntervals() + 1) * Math.pow(2, vs.getNVnodes())));
  }

  // startStates all contain the lowest inveral number 0
  public void testStartState() {
    VirtualStations vs = new VirtualStations();
    Tensor startStates = vs.startStates();
    startStates.forEach(v -> assertEquals(v.Get(0), RealScalar.ZERO));
  }

  public void testActions() {
    VirtualStations vs = new VirtualStations();
    for (Tensor state : vs.states()) {
      Tensor actions = vs.actions(state);
      if (vs.isTerminal(state)) {
        assertEquals(actions.length(), 1);
      } else {
        Scalar expected = Power.of(RealScalar.of(2), Total.of(state.extract(1, state.length())).Get().multiply(RealScalar.of(vs.getNVnodes() - 1)));
        Scalar actual = RealScalar.of(actions.length());
        assertEquals(expected, actual);
      }
    }
  }

  public void testTerminalStates() {
    VirtualStations vs = new VirtualStations();
    for (Tensor state : vs.states()) {
      assertEquals(vs.isTerminal(state), state.Get(0).equals(RealScalar.of(vs.getTimeIntervals())));
    }
  }
}
