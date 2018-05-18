// code fluric
package ch.ethz.idsc.subare.demo.airport;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensors;
import junit.framework.TestCase;

public class AirportTest extends TestCase {
  public void testTerminalState() {
    Airport airport = new Airport();
    assertEquals(airport.isTerminal(Tensors.vector(Airport.LASTT, 0, Airport.VEHICLES)), true);
    assertEquals(airport.actions(Tensors.vector(Airport.LASTT, 0, Airport.VEHICLES)), Tensors.of(RealScalar.ZERO));
    assertEquals(airport.expectedReward(Tensors.vector(Airport.LASTT, 0, Airport.VEHICLES), Tensors.of(RealScalar.ZERO)), RealScalar.ZERO);
    assertEquals(
        airport.reward(Tensors.vector(Airport.LASTT, 0, Airport.VEHICLES), Tensors.of(RealScalar.ZERO), Tensors.vector(Airport.LASTT, 0, Airport.VEHICLES)),
        RealScalar.ZERO);
  }
}
