package ch.ethz.idsc.subare.util;

import ch.ethz.idsc.tensor.io.ObjectFormat;
import junit.framework.TestCase;

public class EnumSizeTest extends TestCase {
  public void testSize() {
    Some s = Some.a;
    byte[] b = ObjectFormat.of(s);
    System.out.println(b.length);
  }
}
