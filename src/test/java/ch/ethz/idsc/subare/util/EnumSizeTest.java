// code by jph
package ch.ethz.idsc.subare.util;

import ch.ethz.idsc.tensor.io.ObjectFormat;
import junit.framework.TestCase;

public class EnumSizeTest extends TestCase {
  enum Some {
    a(3), b(4);
    // ---
    public int asd;
    public int asd3;

    private Some(int a) {
      asd = a;
      asd3 = a + 4;
    }
  }

  public void testSize() {
    Some s = Some.a;
    byte[] b;
    try {
      // System.out.println(Serialization.of(s).length);
      b = ObjectFormat.of(s);
      assertEquals(ObjectFormat.parse(b), s);
    } catch (Exception exception) {
      // ---
    }
  }
}
