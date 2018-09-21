// code by jph
package ch.ethz.idsc.subare.ch06.windy;

import ch.ethz.idsc.subare.core.td.SarsaType;
import junit.framework.TestCase;

public class Sarsa_WindygridTest extends TestCase {
  public void testSimple() throws Exception {
    for (SarsaType sarsaType : SarsaType.values())
      Sarsa_Windygrid.handle(sarsaType, 10);
  }
}
